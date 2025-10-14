#!/usr/bin/env python3
"""
Two-stage decoder for watermarked audio using sync-assisted search.

Master Mode (for known watermarked audio):
- Directly walk through placements and reconstruct bits
- No Stage-A needed, no sync correlation
- Use exact same logic as new_inference.py

Search Mode (for unknown/mixed audio):
- Stage A: sync scan to localize and recover source-second indices
- Stage B: targeted decode using recovered indices
- Enforce index continuity in segment formation

Inputs:
- Audio file (e.g., sampled.wav or watermarked master)
- Slots JSON produced by encode.py (contains sync spec and placements)
- Decode checkpoint (models weights)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.payload_codec import unpack_fields
from pipeline.ingest_and_chunk import (
    rs_decode_167_125,
    rs_encode_167_125,
    deinterleave_bytes,
    interleave_bytes,
)


TARGET_SR = 22050
SEC = 1.0
WIN_S = int(TARGET_SR * SEC)


@dataclass
class Detection:
    """Structured detection result"""
    start_s: float
    end_s: float
    confidence: float
    ber: float
    rs_ok: bool
    payload_snippet: str
    payload_hex: str
    source_sec_offset: int
    num_windows: int
    agreeing_windows: int
    verified: bool = False


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    if sr == TARGET_SR:
        return wav, sr
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav, TARGET_SR


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    """Load audio as mono, preserving original characteristics"""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav, sr = _resample_if_needed(wav, sr)
    # No normalization - preserve original levels
    return wav, sr


def _chunk_audio_1s(wav: torch.Tensor) -> List[torch.Tensor]:
    """Chunk audio into 1-second segments"""
    T = wav.size(-1)
    chunks = []
    cursor = 0
    while cursor < T:
        end = min(cursor + WIN_S, T)
        ch = wav[..., cursor:end]
        if ch.size(-1) < WIN_S:
            ch = torch.nn.functional.pad(ch, (0, WIN_S - ch.size(-1)))
        chunks.append(ch)
        cursor += WIN_S
    if len(chunks) == 0:
        chunks = [torch.nn.functional.pad(wav[..., :0], (0, WIN_S))]
    return chunks


def _bytes_to_bits(by: bytes) -> List[int]:
    bits: List[int] = []
    for b in by:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


def _bits_to_bytes(bits: List[int], bit_order: str = "LSB-first-per-byte") -> bytes:
    """Convert bits to bytes respecting the bit order from slots_map"""
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        if bit_order == "LSB-first-per-byte":
            # LSB first (bit 0 is least significant)
            for k in range(8):
                if i + k < len(bits):
                    b |= ((bits[i + k] & 1) << k)
        elif bit_order == "MSB-first-per-byte":
            # MSB first (bit 0 is most significant)
            for k in range(8):
                if i + k < len(bits):
                    b |= ((bits[i + k] & 1) << (7 - k))
        else:
            # Default to LSB-first for backward compatibility
            for k in range(8):
                if i + k < len(bits):
                    b |= ((bits[i + k] & 1) << k)
        by.append(b)
    return bytes(by)


def _correlate_sync_with_source(model: INNWatermarker, x: torch.Tensor, n_fft: int, hop: int,
                               placements: List[Dict]) -> Tuple[float, int]:
    """
    Correlate with source sync code using per-second sync bins and return both score and estimated source second index.
    """
    X = model.stft(x)  # [1,2,F,T] - sync on imaginary channel (index 1)
    Fbins, Tframes = X.shape[-2], X.shape[-1]

    best_score = 0.0
    best_sec_idx = -1

    # Test against each source second's sync configuration
    for test_sec_idx, sec_data in enumerate(placements):
        if "sync" not in sec_data or "bins" not in sec_data["sync"] or "code" not in sec_data["sync"]:
            continue
            
        sync_bins = sec_data["sync"]["bins"]
        sync_code = sec_data["sync"]["code"]
        
        if len(sync_bins) == 0 or len(sync_code) == 0:
            continue

        # Extract values from this second's sync bins
        vals = []
        for (f, t) in sync_bins:
            if 0 <= f < Fbins and 0 <= t < Tframes:
                # Read from imaginary channel (sync channel)
                vals.append(float(X[0, 1, f, t].item()))
            else:
                vals.append(0.0)  # Out of bounds bins

        if len(vals) == 0 or len(vals) != len(sync_code):
            continue

        # Normalize correlation (scale-invariant)
        v = torch.tensor(vals, dtype=torch.float32)
        c = torch.tensor(sync_code, dtype=torch.float32)
        v_norm = (v - v.mean()) / (v.std() + 1e-6)
        c_norm = (c - c.mean()) / (c.std() + 1e-6)
        score = float(torch.dot(v_norm, c_norm) / max(1, len(vals)))

        if score > best_score:
            best_score = score
            best_sec_idx = test_sec_idx

    return best_score, best_sec_idx


def master_mode_decode(model: INNWatermarker, wav: torch.Tensor, placements: List[Dict], 
                      coded_bits: int, coded_bytes: int, interleave_depth: int, 
                      bit_order: str, expected_checksum: str) -> Detection:
    """
    Master mode: directly walk through placements and reconstruct bits (like new_inference.py)
    """
    chunks = _chunk_audio_1s(wav)
    
    # Use the EXACT same method as the encoder: global bit indexing
    all_bits = [0] * coded_bits
    bit_counts = [0] * coded_bits
    
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx >= len(placements):
            break
            
        x = chunk.to(next(model.parameters()).device).unsqueeze(0)
        M_rec = model.decode(x)  # [1,2,F,T]
        
        # Get placements for this chunk
        chunk_placements = placements[chunk_idx].get("payload", [])
        
        # Use global bit indexing like the encoder does
        for (f, t, amp, bit_idx) in chunk_placements:
            if 0 <= bit_idx < coded_bits:
                val = float(M_rec[0, 0, f, t].item())  # Payload on real channel
                bit = 1 if val >= 0.0 else 0
                all_bits[bit_idx] += bit
                bit_counts[bit_idx] += 1
    
    # Convert to final bits by majority vote (only where we have votes)
    final_bits = []
    for i in range(coded_bits):
        if bit_counts[i] > 0:
            final_bits.append(1 if all_bits[i] > bit_counts[i] / 2 else 0)
        else:
            # Don't default to 0 - this bit was never placed
            final_bits.append(0)
    
    # Convert to bytes and decode
    byte_stream = _bits_to_bytes(final_bits, bit_order)
    if len(byte_stream) < coded_bytes:
        byte_stream = byte_stream + b'\x00' * (coded_bytes - len(byte_stream))
    elif len(byte_stream) > coded_bytes:
        byte_stream = byte_stream[:coded_bytes]
        
    deint = deinterleave_bytes(byte_stream, interleave_depth)
    
    # RS decode
    rs_ok = False
    recovered_payload = None
    try:
        payload = rs_decode_167_125(deint)
        recovered_payload = payload[:125]  # First 125 bytes
        rs_ok = True
    except Exception as e:
        print(f"RS decode failed: {e}")
        recovered_payload = None

    # Verify checksum
    verified = False
    if recovered_payload and expected_checksum:
        try:
            import hashlib
            rs_code = rs_encode_167_125(recovered_payload)
            interleaved = interleave_bytes(rs_code, interleave_depth)
            actual_checksum = hashlib.sha256(interleaved).hexdigest()
            verified = (actual_checksum == expected_checksum)
            print(f"Checksum verification: expected={expected_checksum[:16]}..., actual={actual_checksum[:16]}..., verified={verified}")
        except Exception as e:
            print(f"Checksum verification failed: {e}")

    # Calculate BER
    ber = 0.0
    if recovered_payload:
        try:
            rs_code = rs_encode_167_125(recovered_payload)
            interleaved = interleave_bytes(rs_code, interleave_depth)
            expected_bits = _bytes_to_bits(interleaved)
            if len(expected_bits) == len(final_bits):
                errors = sum(1 for a, b in zip(expected_bits, final_bits) if a != b)
                ber = errors / len(expected_bits)
        except Exception:
            ber = 1.0

    # Create detection
    payload_hex = recovered_payload.hex() if recovered_payload else ""
    
    # Try to unpack the fields using payload_codec
    text = ""
    if recovered_payload:
        try:
            # Define the expected field order (same as encoder)
            field_order = ["ISRC", "ISWC", "Dur", "RDate"]
            unpacked_fields = unpack_fields(recovered_payload, field_order)
            
            # Format as key=value pairs (same as original metadata)
            text_parts = []
            for key, value in unpacked_fields.items():
                text_parts.append(f"{key}={value}")
            text = ";".join(text_parts)
        except Exception as e:
            # Fallback to raw hex if unpacking fails
            text = recovered_payload.hex()

    return Detection(
        start_s=0.0,
        end_s=float(len(chunks)),
        confidence=1.0 if rs_ok else 0.0,
        ber=ber,
        rs_ok=rs_ok,
        payload_snippet=text[:256],
        payload_hex=payload_hex,
        source_sec_offset=0,
        num_windows=len(chunks),
        agreeing_windows=len(chunks) if rs_ok else 0,
        verified=verified
    )


def search_mode_decode(model: INNWatermarker, wav: torch.Tensor, placements: List[Dict],
                      coded_bits: int, coded_bytes: int, interleave_depth: int,
                      bit_order: str, expected_checksum: str, n_fft: int, hop: int,
                      sync_threshold: float, hysteresis_low: float) -> List[Detection]:
    """
    Search mode: Stage-A sync scan + Stage-B targeted decode
    """
    # Stage A: sync scan
    hop_samp = max(1, int(0.5 * TARGET_SR))  # 0.5s hop
    starts = list(range(0, max(1, wav.size(-1) - WIN_S + 1), hop_samp))
    sync_results: List[Tuple[float, int, int]] = []  # (score, start_sample, source_sec_idx)

    print(f"Stage A: Scanning {len(starts)} windows with per-second sync...")
    for st in starts:
        win = wav[:, st:st + WIN_S]
        if win.size(-1) < WIN_S:
            win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
        
        x = win.to(model.stft.win_length.device).unsqueeze(0)
        score, source_sec_idx = _correlate_sync_with_source(model, x, n_fft, hop, placements)
        sync_results.append((score, st, source_sec_idx))

    # Build segments with index continuity
    segments = []
    current_segment = None
    
    for score, start_sample, source_sec_idx in sync_results:
        if score >= sync_threshold:
            if current_segment is None:
                current_segment = (start_sample, start_sample + WIN_S, source_sec_idx)
            else:
                # Check index continuity - allow ±1 jitter
                expected_next = current_segment[2] + 1
                if abs(source_sec_idx - expected_next) <= 1:
                    current_segment = (current_segment[0], start_sample + WIN_S, source_sec_idx)
                else:
                    segments.append(current_segment)
                    current_segment = (start_sample, start_sample + WIN_S, source_sec_idx)
        elif current_segment is not None and score >= hysteresis_low:
            expected_next = current_segment[2] + 1
            if abs(source_sec_idx - expected_next) <= 1:
                current_segment = (current_segment[0], start_sample + WIN_S, source_sec_idx)
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
    
    if current_segment is not None:
        segments.append(current_segment)

    print(f"Stage B: Processing {len(segments)} candidate segments...")
    
    # Stage B: targeted decode
    detections = []
    
    for seg_idx, (seg_s, seg_e, source_sec_offset) in enumerate(segments):
        print(f"  Processing segment {seg_idx + 1}/{len(segments)}: {seg_s}-{seg_e} (source offset: {source_sec_offset})")
        
        # Find nearest Stage-A results for each window in this segment
        window_data = []
        cursor = seg_s
        while cursor < seg_e:
            # Find nearest Stage-A result within ±hop_samp/2
            nearest_score = 0.0
            nearest_source_sec_idx = -1
            for score, start_sample, sec_idx in sync_results:
                if abs(start_sample - cursor) <= hop_samp // 2:
                    if score > nearest_score:
                        nearest_score = score
                        nearest_source_sec_idx = sec_idx
            
            if nearest_source_sec_idx >= 0:
                win = wav[:, cursor:cursor + WIN_S]
                if win.size(-1) < WIN_S:
                    win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
                
                x = win.to(next(model.parameters()).device).unsqueeze(0)
                window_data.append((x, nearest_source_sec_idx))
            
            cursor += WIN_S

        if not window_data:
            print(f"    No valid windows found for segment {seg_idx + 1}")
            continue

        # Collect bits from all windows
        all_bits = [0] * coded_bits
        bit_counts = [0] * coded_bits
        contributing_windows = 0
        
        for x, source_sec_idx in window_data:
            M_rec = model.decode(x)
            
            # Get placements for this source second
            if 0 <= source_sec_idx < len(placements):
                chunk_placements = placements[source_sec_idx].get("payload", [])
                
                for (f, t, amp, bit_idx) in chunk_placements:
                    if 0 <= bit_idx < coded_bits:
                        val = float(M_rec[0, 0, f, t].item())
                        bit = 1 if val >= 0.0 else 0
                        all_bits[bit_idx] += bit
                        bit_counts[bit_idx] += 1
                
                contributing_windows += 1

        # Convert to final bits by majority vote (only where we have votes)
        bits = []
        for i in range(coded_bits):
            if bit_counts[i] > 0:
                bits.append(1 if all_bits[i] > bit_counts[i] / 2 else 0)
            else:
                # Don't default to 0 - skip this bit
                continue

        if not bits:
            print(f"    No valid bits collected for segment {seg_idx + 1}")
            continue

        # Pad to required length if needed
        while len(bits) < coded_bits:
            bits.append(0)

        # RS decode
        byte_stream = _bits_to_bytes(bits, bit_order)
        if len(byte_stream) < coded_bytes:
            byte_stream = byte_stream + b'\x00' * (coded_bytes - len(byte_stream))
        elif len(byte_stream) > coded_bytes:
            byte_stream = byte_stream[:coded_bytes]
            
        deint = deinterleave_bytes(byte_stream, interleave_depth)
        
        rs_ok = False
        recovered_payload = None
        try:
            payload = rs_decode_167_125(deint)
            recovered_payload = payload[:125]
            rs_ok = True
        except Exception as e:
            print(f"    RS decode failed: {e}")

        if not rs_ok:
            print(f"    Segment {seg_idx + 1} failed RS decode, skipping")
            continue

        # Verify checksum
        verified = False
        if expected_checksum:
            try:
                import hashlib
                rs_code = rs_encode_167_125(recovered_payload)
                interleaved = interleave_bytes(rs_code, interleave_depth)
                actual_checksum = hashlib.sha256(interleaved).hexdigest()
                verified = (actual_checksum == expected_checksum)
                print(f"    Checksum verification: expected={expected_checksum[:16]}..., actual={actual_checksum[:16]}..., verified={verified}")
            except Exception as e:
                print(f"    Checksum verification failed: {e}")

        # Calculate BER
        ber = 0.0
        if recovered_payload:
            try:
                rs_code = rs_encode_167_125(recovered_payload)
                interleaved = interleave_bytes(rs_code, interleave_depth)
                expected_bits = _bytes_to_bits(interleaved)
                if len(expected_bits) == len(bits):
                    errors = sum(1 for a, b in zip(expected_bits, bits) if a != b)
                    ber = errors / len(bits)
            except Exception:
                ber = 1.0

        # Create detection
        payload_hex = recovered_payload.hex() if recovered_payload else ""
        
        # Try to unpack the fields using payload_codec
        text = ""
        if recovered_payload:
            try:
                # Define the expected field order (same as encoder)
                field_order = ["ISRC", "ISWC", "Dur", "RDate"]
                unpacked_fields = unpack_fields(recovered_payload, field_order)
                
                # Format as key=value pairs (same as original metadata)
                text_parts = []
                for key, value in unpacked_fields.items():
                    text_parts.append(f"{key}={value}")
                text = ";".join(text_parts)
            except Exception as e:
                # Fallback to raw hex if unpacking fails
                text = recovered_payload.hex()

        detection = Detection(
            start_s=float(seg_s / TARGET_SR),
            end_s=float(seg_e / TARGET_SR),
            confidence=1.0 if rs_ok else 0.0,
            ber=ber,
            rs_ok=rs_ok,
            payload_snippet=text[:256],
            payload_hex=payload_hex,
            source_sec_offset=source_sec_offset,
            num_windows=contributing_windows,
            agreeing_windows=contributing_windows if rs_ok else 0,
            verified=verified
        )
        detections.append(detection)

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="sampled.wav")
    parser.add_argument("--slots", type=str, default="slots_map.json")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "inn_decode_best.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "master", "search"], 
                       help="Decode mode: auto (detect), master (known watermarked), search (unknown audio)")
    parser.add_argument("--sync_threshold", type=float, default=0.4, help="Sync correlation threshold")
    parser.add_argument("--hysteresis_low", type=float, default=0.2, help="Lower threshold for segment continuation")
    parser.add_argument("--out_detections", type=str, default="detections.json", help="Output detections JSON")
    args = parser.parse_args()

    # Load slots config
    with open(args.slots, "r", encoding="utf-8") as f:
        slots_map = json.load(f)
    
    # Validate schema version
    schema_version = slots_map.get("schema_version", "0.0")
    if schema_version != "1.0":
        print(f"Warning: Schema version {schema_version} may not be compatible")
    
    # Read STFT parameters from slots_map (must match encoder)
    stft_cfg = slots_map.get("stft", {"n_fft": 1024, "hop": 512, "win_length": 1024})
    n_fft = int(stft_cfg.get("n_fft", 1024))
    hop = int(stft_cfg.get("hop", 512))
    
    # Read payload specification from slots_map (authoritative)
    payload_spec = slots_map.get("payload_spec", {})
    payload_verification = slots_map.get("payload_verification", {})
    
    payload_bytes = int(payload_spec.get("payload_bytes", 125))
    coded_bytes = int(payload_verification.get("coded_bytes", 167))
    coded_bits = int(payload_verification.get("coded_bits", 1336))
    interleave_depth = int(payload_spec.get("interleave_depth", 4))
    bit_order = payload_spec.get("bit_order", "LSB-first-per-byte")
    
    # Read per-second placements
    placements = slots_map.get("placements", [])
    if not placements:
        raise RuntimeError("No placements found in slots_map.json")
    
    # Read channel assignments and verify they match our usage
    channel_assignments = slots_map.get("channel_assignments", {})
    payload_channel = channel_assignments.get("payload_channel", "real")
    sync_channel = channel_assignments.get("sync_channel", "imaginary")
    
    if payload_channel != "real" or sync_channel != "imaginary":
        print(f"Warning: Channel assignments don't match expected (payload=real, sync=imaginary)")
        print(f"  Found: payload={payload_channel}, sync={sync_channel}")
    
    # Read payload verification data
    payload_verification = slots_map.get("payload_verification", {})
    expected_checksum = payload_verification.get("checksum", "")
    
    # Build model
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(args.device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    # Load audio
    wav, sr = _load_audio_mono(args.audio)
    if sr != TARGET_SR:
        raise RuntimeError("Unexpected SR after resample")

    # Determine mode
    if args.mode == "auto":
        # Auto-detect: if audio length matches expected chunks, use master mode
        expected_chunks = len(placements)
        actual_chunks = (wav.size(-1) + WIN_S - 1) // WIN_S
        mode = "master" if abs(expected_chunks - actual_chunks) <= 1 else "search"
        print(f"Auto-detected mode: {mode} (expected {expected_chunks} chunks, got {actual_chunks})")
    else:
        mode = args.mode

    # Decode based on mode
    if mode == "master":
        print("Using Master Mode: Direct placement walkthrough")
        detection = master_mode_decode(model, wav, placements, coded_bits, coded_bytes, 
                                     interleave_depth, bit_order, expected_checksum)
        detections = [detection]
    else:
        print("Using Search Mode: Stage-A sync scan + Stage-B targeted decode")
        detections = search_mode_decode(model, wav, placements, coded_bits, coded_bytes,
                                      interleave_depth, bit_order, expected_checksum,
                                      n_fft, hop, args.sync_threshold, args.hysteresis_low)

    # Save detections with provenance
    detections_data = {
        "schema_version": "1.0",
        "audio_file": args.audio,
        "slots_file": args.slots,
        "mode": mode,
        "model_id": slots_map.get("model_id", "unknown"),
        "model_hash": slots_map.get("model_hash", "unknown"),
        "stft_config": stft_cfg,
        "total_detections": len(detections),
        "detections": [
            {
                "start_s": d.start_s,
                "end_s": d.end_s,
                "confidence": d.confidence,
                "ber": d.ber,
                "rs_ok": d.rs_ok,
                "payload_snippet": d.payload_snippet,
                "payload_hex": d.payload_hex,
                "source_sec_offset": d.source_sec_offset,
                "num_windows": d.num_windows,
                "agreeing_windows": d.agreeing_windows,
                "verified": d.verified
            }
            for d in detections
        ]
    }

    with open(args.out_detections, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2)

    # Print summary
    print(f"\n=== Detection Summary ===")
    print(f"Mode: {mode}")
    print(f"Total detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"Detection {i+1}:")
        print(f"  Time: {det.start_s:.2f}s - {det.end_s:.2f}s")
        print(f"  Confidence: {det.confidence:.3f}")
        print(f"  BER: {det.ber:.3f}")
        print(f"  RS OK: {det.rs_ok}")
        print(f"  Verified: {det.verified}")
        print(f"  Payload (hex): {det.payload_hex[:64]}{'...' if len(det.payload_hex) > 64 else ''}")
        if det.payload_snippet:
            print(f"  Payload (text): {det.payload_snippet}")
        print(f"  Source offset: {det.source_sec_offset}")
        print(f"  Windows: {det.agreeing_windows}/{det.num_windows}")
        print()

    print(f"Detections saved to: {args.out_detections}")


if __name__ == "__main__":
    main()