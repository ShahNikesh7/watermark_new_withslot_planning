#!/usr/bin/env python3
"""
Two-stage decoder for watermarked audio using sync-assisted search.

Stage A (sync scan, DSP-only):
- Downmix to mono and resample to 22.05 kHz
- Slide 1s windows (hop 0.5s) over audio
- For each window, compute STFT and read imaginary parts at per-second sync bins
- Correlate with stored per-second sync codes to get score and source second index
- Build segments from contiguous high-sync windows using threshold/hysteresis

Stage B (targeted NN decode):
- For each candidate segment, process 1s subwindows
- Run INNWatermarker.decode on each second
- Read payload symbols from encoder's exact placements (no fallback allocation)
- Use per-second placements and respect encoder's bit accounting
- Weighted voting with encoder amplitudes and local SNR
- RS decode with proper gating and BER calculation

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


def _slide_indices(T: int, hop: int) -> List[int]:
    idxs = list(range(0, max(1, T - WIN_S + 1), hop))
    if len(idxs) == 0:
        idxs = [0]
    return idxs


def _correlate_sync_with_source(model: INNWatermarker, x: torch.Tensor, n_fft: int, hop: int,
                               placements: List[Dict]) -> Tuple[float, int]:
    """
    Correlate with source sync code using per-second sync bins and return both score and estimated source second index.

    Args:
        model: INN model for STFT
        x: [1,1,T] 1s window
        n_fft: STFT n_fft parameter
        hop: STFT hop parameter
        placements: List of per-second placement data from slots_map

    Returns:
        (correlation_score, estimated_source_sec_idx)
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


def _bytes_to_bits(by: bytes) -> List[int]:
    bits: List[int] = []
    for b in by:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for k in range(8):
            if i + k < len(bits):
                b |= ((bits[i + k] & 1) << k)
        by.append(b)
    return bytes(by)


def _calculate_local_snr(X: torch.Tensor, f: int, t: int, window_size: int = 3) -> float:
    """Calculate local SNR around a frequency-time bin"""
    Fbins, Tframes = X.shape[-2], X.shape[-1]
    
    # Define local window around the bin
    f_start = max(0, f - window_size // 2)
    f_end = min(Fbins, f + window_size // 2 + 1)
    t_start = max(0, t - window_size // 2)
    t_end = min(Tframes, t + window_size // 2 + 1)
    
    # Extract local region
    local_region = X[0, 0, f_start:f_end, t_start:t_end]
    
    # Calculate signal power (magnitude squared)
    signal_power = local_region.abs().pow(2).mean().item()
    
    # Estimate noise power from surrounding region
    noise_region = X[0, 0, :, :].clone()
    noise_region[f_start:f_end, t_start:t_end] = 0  # Remove signal region
    noise_power = noise_region.abs().pow(2).mean().item()
    
    # Calculate SNR in dB
    if noise_power > 1e-10:
        snr_db = 10.0 * math.log10(signal_power / noise_power)
        return max(0.0, snr_db)  # Clamp to positive values
    else:
        return 30.0  # High SNR if noise is very low


def _build_segments_from_sync(sync_results: List[Tuple[float, int, int]], 
                             threshold: float, 
                             hysteresis_low: float) -> List[Tuple[int, int, int]]:
    """Build segments from contiguous high-sync windows using threshold/hysteresis"""
    if not sync_results:
        return []
    
    # Sort by start position
    sync_results.sort(key=lambda x: x[1])
    
    segments = []
    current_segment = None
    
    for score, start_sample, source_sec_idx in sync_results:
        if score >= threshold:
            if current_segment is None:
                # Start new segment
                current_segment = (start_sample, start_sample + WIN_S, source_sec_idx)
            else:
                # Extend current segment if close enough
                if start_sample - current_segment[1] <= int(0.75 * TARGET_SR):
                    current_segment = (current_segment[0], start_sample + WIN_S, source_sec_idx)
                else:
                    # Close current segment and start new one
                    segments.append(current_segment)
                    current_segment = (start_sample, start_sample + WIN_S, source_sec_idx)
        elif current_segment is not None and score >= hysteresis_low:
            # Continue segment with lower threshold
            if start_sample - current_segment[1] <= int(0.75 * TARGET_SR):
                current_segment = (current_segment[0], start_sample + WIN_S, source_sec_idx)
        else:
            # Close current segment
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
    
    # Close final segment
    if current_segment is not None:
        segments.append(current_segment)
    
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="sampled.wav")
    parser.add_argument("--slots", type=str, default="slots_map.json")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "inn_decode_best.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--sync_threshold", type=float, default=0.3, help="Sync correlation threshold")
    parser.add_argument("--hysteresis_low", type=float, default=0.15, help="Lower threshold for segment continuation")
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
    
    # Read payload specification from slots_map
    payload_spec = slots_map.get("payload_spec", {})
    payload_bytes = int(payload_spec.get("payload_bytes", 125))
    coded_bytes = int(payload_spec.get("coded_bytes", 167))
    coded_bits = int(payload_spec.get("coded_bits", 1336))
    interleave_depth = int(payload_spec.get("interleave_depth", 4))
    bit_order = payload_spec.get("bit_order", "LSB-first-per-byte")
    
    # Read per-second placements
    placements = slots_map.get("placements", [])
    if not placements:
        raise RuntimeError("No placements found in slots_map.json")
    
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

    # Handle edge cases
    if wav.size(-1) < WIN_S:
        print("Warning: Audio shorter than 1 second, padding to 1 second")
        wav = torch.nn.functional.pad(wav, (0, WIN_S - wav.size(-1)))

    # Stage A: sync scan using per-second sync bins from slots_map
    hop_samp = max(1, int(args.hop_sec * TARGET_SR))
    starts = _slide_indices(wav.size(-1), hop=hop_samp)
    sync_results: List[Tuple[float, int, int]] = []  # (score, start_sample, source_sec_idx)

    print(f"Stage A: Scanning {len(starts)} windows with per-second sync...")
    for i, st in enumerate(starts):
        win = wav[:, st:st + WIN_S]
        if win.size(-1) < WIN_S:
            win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
        
        x = win.to(args.device).unsqueeze(0)
        
        # Use per-second sync correlation
        score, source_sec_idx = _correlate_sync_with_source(model, x, n_fft, hop, placements)
        sync_results.append((score, st, source_sec_idx))

    # Build segments using threshold/hysteresis
    segments = _build_segments_from_sync(sync_results, args.sync_threshold, args.hysteresis_low)
    print(f"Stage B: Processing {len(segments)} candidate segments...")
    
    # Stage B: targeted NN decode around segments
    detections: List[Detection] = []

    # Helper: read placements for a given source second index
    def get_source_placements(source_sec_idx: int) -> List[Tuple[int, int, int, float]]:
        """Get placements for source second index: (f, t, bit_idx, amplitude)"""
        if 0 <= source_sec_idx < len(placements):
            ent = placements[source_sec_idx]
            arr = ent.get("payload", [])
            out = []
            for it in arr:
                if isinstance(it, list) and len(it) >= 4:
                    f, t, amp, bit_idx = int(it[0]), int(it[1]), float(it[2]), int(it[3])
                    out.append((f, t, bit_idx, amp))
            return out
        return []

    for seg_idx, (seg_s, seg_e, source_sec_offset) in enumerate(segments):
        print(f"  Processing segment {seg_idx + 1}/{len(segments)}: {seg_s}-{seg_e} (source offset: {source_sec_offset})")
        
        # Collect votes with weighted voting using encoder amplitudes and local SNR
        all_votes: Dict[int, List[Tuple[int, float]]] = {}  # bit_idx -> [(vote, weight), ...]
        window_count = 0
        
        cursor = seg_s
        while cursor < seg_e:
            win = wav[:, cursor:cursor + WIN_S]
            if win.size(-1) < WIN_S:
                win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
            
            x = win.to(args.device).unsqueeze(0)
            M_rec = model.decode(x)  # [1,2,F,T]
            X_stft = model.stft(x)  # For SNR calculation
            
            # Calculate source second index for this window
            # The source_sec_offset is the detected source second from sync correlation
            # We need to map the current window position to the corresponding source second
            window_offset_seconds = (cursor - seg_s) // WIN_S
            window_source_sec = source_sec_offset + window_offset_seconds
            
            # Clamp to available placements range
            max_source_sec = len(placements) - 1
            window_source_sec = max(0, min(window_source_sec, max_source_sec))
            pl = get_source_placements(window_source_sec)
            
            if not pl:
                # Skip window if no placements found (no fallback allocation)
                print(f"    Warning: No placements found for source second {window_source_sec}, skipping window")
                cursor += WIN_S
                continue

            for (f, t, bit_idx, amp_weight) in pl:
                if bit_idx >= coded_bits:
                    continue  # Skip out-of-bounds bit indices
                    
                val = float(M_rec[0, 0, f, t].item())  # Payload on real channel
                bit = 1 if val >= 0.0 else 0
                
                # Calculate local SNR for this bin
                local_snr = _calculate_local_snr(X_stft, f, t)
                snr_weight = min(1.0, local_snr / 20.0)  # Normalize SNR to 0-1 range
                
                # Combined weight: encoder amplitude × local SNR
                combined_weight = abs(amp_weight) * snr_weight
                
                all_votes.setdefault(bit_idx, []).append((bit, combined_weight))
            
            window_count += 1
            cursor += WIN_S

        if not all_votes:
            print(f"    No valid votes collected for segment {seg_idx + 1}")
            continue

        # Weighted majority vote per bit index
        bits: List[int] = []
        bit_weights: List[float] = []
        
        for i in range(coded_bits):
            votes = all_votes.get(i, [])
            if votes:
                # Weighted voting
                total_weight = sum(weight for _, weight in votes)
                weighted_ones = sum(weight for bit, weight in votes if bit == 1)
                bit = 1 if weighted_ones >= total_weight / 2.0 else 0
                bits.append(bit)
                bit_weights.append(total_weight)
            else:
                bits.append(0)  # Default to 0 for missing bits
                bit_weights.append(0.0)

        # Calculate confidence from voting margins
        margins = []
        for i in range(len(bits)):
            votes = all_votes.get(i, [])
            if votes:
                total_weight = sum(weight for _, weight in votes)
                weighted_ones = sum(weight for bit, weight in votes if bit == 1)
                p = weighted_ones / max(1e-6, total_weight)
                margins.append(abs(p - 0.5) * 2.0)
        
        confidence = float(sum(margins) / max(1, len(margins))) if margins else 0.0

        # RS decode with proper gating
        byte_stream = _bits_to_bytes(bits)
        if len(byte_stream) < coded_bytes:
            byte_stream = byte_stream + b'\x00' * (coded_bytes - len(byte_stream))
        elif len(byte_stream) > coded_bytes:
            byte_stream = byte_stream[:coded_bytes]
            
        deint = deinterleave_bytes(byte_stream, interleave_depth)
        rs_ok = False
        recovered_payload: Optional[bytes] = None
        
        try:
            payload = rs_decode_167_125(deint)
            recovered_payload = payload[:payload_bytes]
            rs_ok = True
        except Exception as e:
            print(f"    RS decode failed: {e}")
            recovered_payload = None

        # Only proceed if RS decode succeeded
        if not rs_ok or recovered_payload is None:
            print(f"    Segment {seg_idx + 1} failed RS decode, skipping")
            continue

        # Calculate BER against re-encoded payload
        ber = 0.0
        try:
            rs_code = rs_encode_167_125(recovered_payload)
            interleaved = interleave_bytes(rs_code, interleave_depth)
            expected_bits = _bytes_to_bits(interleaved)
            if len(expected_bits) == len(bits):
                errors = sum(1 for a, b in zip(expected_bits, bits) if a != b)
                ber = errors / len(bits)
        except Exception:
            ber = 1.0

        # Calculate agreeing windows
        agreeing_windows = 0
        try:
            rs_code = rs_encode_167_125(recovered_payload)
            interleaved = interleave_bytes(rs_code, interleave_depth)
            expected_bits = _bytes_to_bits(interleaved)
            
            cursor = seg_s
            while cursor < seg_e:
                win = wav[:, cursor:cursor + WIN_S]
                if win.size(-1) < WIN_S:
                    win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
                
                x = win.to(args.device).unsqueeze(0)
                M_rec = model.decode(x)
                
                window_offset_seconds = (cursor - seg_s) // WIN_S
                window_source_sec = source_sec_offset + window_offset_seconds
                
                # Clamp to available placements range
                max_source_sec = len(placements) - 1
                window_source_sec = max(0, min(window_source_sec, max_source_sec))
                pl = get_source_placements(window_source_sec)
                
                if not pl:
                    cursor += WIN_S
                    continue

                window_agrees = True
                for (f, t, bit_idx, _) in pl:
                    if bit_idx < len(expected_bits):
                        val = float(M_rec[0, 0, f, t].item())
                        bit = 1 if val >= 0.0 else 0
                        if bit != expected_bits[bit_idx]:
                            window_agrees = False
                            break
                
                if window_agrees:
                    agreeing_windows += 1
                
                cursor += WIN_S
        except Exception:
            pass

        # Verify payload if checksum is available
        verified = False
        if expected_checksum:
            try:
                rs_code = rs_encode_167_125(recovered_payload)
                interleaved = interleave_bytes(rs_code, interleave_depth)
                import hashlib
                actual_checksum = hashlib.sha256(interleaved).hexdigest()
                verified = (actual_checksum == expected_checksum)
            except Exception:
                pass

        # Create detection
        try:
            text = recovered_payload.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

        # Use sync-based boundaries with single NN confirmation
        start_s = float(seg_s / TARGET_SR)
        end_s = float(seg_e / TARGET_SR)
        
        # Confirm edges with single NN pass
        edge_confirmed_start = start_s
        edge_confirmed_end = end_s
        
        # Check start edge
        if seg_s >= WIN_S:
            edge_win = wav[:, seg_s - WIN_S:seg_s]
            if edge_win.size(-1) == WIN_S:
                edge_x = edge_win.to(args.device).unsqueeze(0)
                edge_score, _ = _correlate_sync_with_source(model, edge_x, n_fft, hop, placements)
                if edge_score >= args.hysteresis_low:
                    edge_confirmed_start = float((seg_s - WIN_S) / TARGET_SR)
        
        # Check end edge
        if seg_e + WIN_S <= wav.size(-1):
            edge_win = wav[:, seg_e:seg_e + WIN_S]
            if edge_win.size(-1) == WIN_S:
                edge_x = edge_win.to(args.device).unsqueeze(0)
                edge_score, _ = _correlate_sync_with_source(model, edge_x, n_fft, hop, placements)
                if edge_score >= args.hysteresis_low:
                    edge_confirmed_end = float((seg_e + WIN_S) / TARGET_SR)

        detection = Detection(
            start_s=edge_confirmed_start,
            end_s=edge_confirmed_end,
            confidence=confidence,
            ber=ber,
            rs_ok=rs_ok,
            payload_snippet=text[:256],
            source_sec_offset=source_sec_offset,
            num_windows=window_count,
            agreeing_windows=agreeing_windows,
            verified=verified
        )
        detections.append(detection)

    # Merge detections that are close together
    merged_detections = []
    if detections:
        detections.sort(key=lambda d: d.start_s)
        current = detections[0]
        
        for next_det in detections[1:]:
            if next_det.start_s - current.end_s <= 0.5:  # Within 0.5 seconds
                # Merge detections
                current = Detection(
                    start_s=current.start_s,
                    end_s=max(current.end_s, next_det.end_s),
                    confidence=max(current.confidence, next_det.confidence),
                    ber=min(current.ber, next_det.ber),
                    rs_ok=current.rs_ok or next_det.rs_ok,
                    payload_snippet=current.payload_snippet if current.confidence >= next_det.confidence else next_det.payload_snippet,
                    source_sec_offset=current.source_sec_offset,
                    num_windows=current.num_windows + next_det.num_windows,
                    agreeing_windows=current.agreeing_windows + next_det.agreeing_windows,
                    verified=current.verified or next_det.verified
                )
            else:
                merged_detections.append(current)
                current = next_det
        
        merged_detections.append(current)

    # Save detections with provenance
    detections_data = {
        "schema_version": "1.0",
        "audio_file": args.audio,
        "slots_file": args.slots,
        "model_id": slots_map.get("model_id", "unknown"),
        "model_hash": slots_map.get("model_hash", "unknown"),
        "stft_config": stft_cfg,
        "total_detections": len(merged_detections),
        "detections": [
            {
                "start_s": d.start_s,
                "end_s": d.end_s,
                "confidence": d.confidence,
                "ber": d.ber,
                "rs_ok": d.rs_ok,
                "payload_snippet": d.payload_snippet,
                "source_sec_offset": d.source_sec_offset,
                "num_windows": d.num_windows,
                "agreeing_windows": d.agreeing_windows,
                "verified": d.verified
            }
            for d in merged_detections
        ]
    }

    with open(args.out_detections, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2)

    # Print summary
    print(f"\n=== Detection Summary ===")
    print(f"Total detections: {len(merged_detections)}")
    for i, det in enumerate(merged_detections):
        print(f"Detection {i+1}:")
        print(f"  Time: {det.start_s:.2f}s - {det.end_s:.2f}s")
        print(f"  Confidence: {det.confidence:.3f}")
        print(f"  BER: {det.ber:.3f}")
        print(f"  RS OK: {det.rs_ok}")
        print(f"  Verified: {det.verified}")
        print(f"  Payload: {det.payload_snippet}")
        print(f"  Source offset: {det.source_sec_offset}")
        print(f"  Windows: {det.agreeing_windows}/{det.num_windows}")
        print()

    print(f"Detections saved to: {args.out_detections}")


if __name__ == "__main__":
    main()