#!/usr/bin/env python3
"""
Two-stage decoder for watermarked audio using sync-assisted search.

Stage A (sync scan, DSP-only):
- Downmix to mono and resample to 22.05 kHz
- Slide 1s windows (hop 0.5s) over audio
- For each window, compute STFT and read real parts at reserved sync bins
- Correlate with known sync code (Barker-13 tiled) to get a score
- Keep top candidate windows; cluster contiguous hits into segments

Stage B (targeted NN decode):
- For each candidate segment ±1s margin, process 1s subwindows
- Run INNWatermarker.decode on each second
- Gather payload symbols at placements from slots map (preferred) or allocate
- Vote across repetitions, deinterleave + RS decode

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
    allocate_slots_and_amplitudes,
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


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    if sr == TARGET_SR:
        return wav, sr
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav, TARGET_SR


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav, sr = _resample_if_needed(wav, sr)
    # normalize for stable correlation
    peak = wav.abs().max().item()
    if peak > 0:
        wav = wav / peak
    return wav, sr


def _slide_indices(T: int, hop: int) -> List[int]:
    idxs = list(range(0, max(1, T - WIN_S + 1), hop))
    if len(idxs) == 0:
        idxs = [0]
    return idxs


def _build_sync_bins(Fbins: int, Tframes: int, sr: int, n_fft: int, K_freq: int, K_time: int, f_low_hz: float, f_high_hz: float) -> List[Tuple[int, int]]:
    f_max = sr / 2.0
    low_bin = max(1, int((f_low_hz / f_max) * (n_fft // 2)))
    high_bin = min(n_fft // 2, int((f_high_hz / f_max) * (n_fft // 2)))
    if high_bin <= low_bin:
        low_bin, high_bin = 1, max(2, n_fft // 4)
    if K_freq <= 1:
        freq_bins = [max(1, (low_bin + high_bin) // 2)]
    else:
        step = (high_bin - low_bin) / float(K_freq + 1)
        freq_bins = [int(low_bin + (i + 1) * step) for i in range(K_freq)]
    if K_time <= 1:
        time_frames = [max(0, (Tframes - 1) // 2)]
    else:
        step_t = max(1.0, (Tframes - 1) / float(K_time + 1))
        time_frames = [int(round((i + 1) * step_t)) for i in range(K_time)]
        time_frames = [min(tf, Tframes - 1) for tf in time_frames]
    bins: List[Tuple[int,int]] = []
    for ti in time_frames:
        for fi in freq_bins:
            bins.append((fi, ti))
    return bins


def _barker13() -> List[int]:
    seq01 = [1,1,1,1,1,0,0,1,1,0,1,0,1]
    return [1 if v == 1 else -1 for v in seq01]


def _correlate_sync_with_source(model: INNWatermarker, x: torch.Tensor, n_fft: int, hop: int, 
                               sync_bins: List[Tuple[int, int]], sync_code: List[int]) -> Tuple[float, int]:
    """
    Correlate with source sync code and return both score and estimated source second index.
    
    Args:
        model: INN model for STFT
        x: [1,1,T] 1s window
        n_fft: STFT n_fft parameter
        hop: STFT hop parameter
        sync_bins: List of (f, t) tuples for sync bins
        sync_code: Source sync code vector
    
    Returns:
        (correlation_score, estimated_source_sec_idx)
    """
    # x: [1,1,T] 1s window
    X = model.stft(x)  # [1,2,F,T]
    Fbins, Tframes = X.shape[-2], X.shape[-1]
    
    # Use provided sync bins and code from source
    K = len(sync_bins)
    if K == 0:
        return 0.0, -1
    
    # Use imaginary part for BPSK readout (sync is on channel 1)
    vals = []
    for (f, t) in sync_bins:
        if 0 <= f < Fbins and 0 <= t < Tframes:
            vals.append(float(X[0, 1, f, t].item()))
        else:
            vals.append(0.0)  # Out of bounds bins
    
    if len(vals) == 0:
        return 0.0, -1
    
    # Try multiple phase shifts to find the best match and recover source second index
    base_code = _barker13()
    best_score = 0.0
    best_sec_idx = -1
    
    # Test different phase shifts (source second indices)
    for test_sec_idx in range(min(100, len(sync_code))):  # Limit search space
        # Generate expected code for this source second index
        if test_sec_idx < len(sync_code):
            expected_code = sync_code[test_sec_idx]
        else:
            # Fallback to rotation-based code if not in stored sync
            code = (base_code * ((K + len(base_code) - 1) // len(base_code)))[:K]
            if K > 0 and test_sec_idx % K != 0:
                r = test_sec_idx % K
                code = code[r:] + code[:r]
            expected_code = code
        
        if len(expected_code) != len(vals):
            continue
            
        # Normalized correlation
        v = torch.tensor(vals, dtype=torch.float32)
        c = torch.tensor(expected_code, dtype=torch.float32)
        v_norm = (v - v.mean()) / (v.std() + 1e-6)
        c_norm = (c - c.mean()) / (c.std() + 1e-6)
        score = float(torch.dot(v_norm, c_norm) / max(1, len(vals)))
        
        if score > best_score:
            best_score = score
            best_sec_idx = test_sec_idx
    
    return best_score, best_sec_idx


def _correlate_sync(model: INNWatermarker, x: torch.Tensor, n_fft: int, hop: int, sync_cfg: Dict, sec_index: int) -> float:
    """Legacy sync correlation function - kept for backward compatibility"""
    # x: [1,1,T] 1s window
    X = model.stft(x)  # [1,2,F,T]
    Fbins, Tframes = X.shape[-2], X.shape[-1]
    bins = _build_sync_bins(Fbins, Tframes, sr=TARGET_SR, n_fft=n_fft,
                            K_freq=int(sync_cfg["k_freq"]), K_time=int(sync_cfg["k_time"]),
                            f_low_hz=float(sync_cfg["f_low_hz"]), f_high_hz=float(sync_cfg["f_high_hz"]))
    K = len(bins)
    base = _barker13()
    code = (base * ((K + len(base) - 1) // len(base)))[:K]
    if K > 0 and sec_index % K != 0:
        r = sec_index % K
        code = code[r:] + code[:r]
    # Use imaginary part for BPSK readout (sync is on channel 1)
    vals = []
    for (f, t) in bins:
        vals.append(float(X[0, 1, f, t].item()))
    if len(vals) == 0:
        return 0.0
    # normalized correlation
    v = torch.tensor(vals, dtype=torch.float32)
    c = torch.tensor(code, dtype=torch.float32)
    v = (v - v.mean()) / (v.std() + 1e-6)
    c = (c - c.mean()) / (c.std() + 1e-6)
    score = float(torch.dot(v, c) / max(1, len(vals)))
    return score


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="sampled.wav")
    parser.add_argument("--slots", type=str, default="slots_map.json")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "inn_decode_best.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=8, help="Keep top-K sync windows for Stage B")
    parser.add_argument("--out_detections", type=str, default="detections.json", help="Output detections JSON")
    args = parser.parse_args()

    # Load slots config
    with open(args.slots, "r", encoding="utf-8") as f:
        slots_map = json.load(f)
    
    # Validate schema version
    schema_version = slots_map.get("schema_version", "0.0")
    if schema_version != "1.0":
        print(f"Warning: Schema version {schema_version} may not be compatible")
    
    stft_cfg = slots_map.get("stft", {"n_fft": 1024, "hop": 512, "win_length": 1024})
    n_fft = int(stft_cfg.get("n_fft", 1024))
    hop = int(stft_cfg.get("hop", 512))
    sync_cfg = slots_map.get("sync_spec", {})
    payload_cfg = slots_map.get("payload_spec", {})
    repeat = int(slots_map.get("repeat", 1))
    placements = slots_map.get("placements", [])
    psychoacoustic_params = slots_map.get("psychoacoustic_params", {})

    # Build model
    state = torch.load(args.ckpt, map_location=args.device)
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

    # Stage A: sync scan using source sync from slots_map
    hop_samp = max(1, int(args.hop_sec * TARGET_SR))
    starts = _slide_indices(wav.size(-1), hop=hop_samp)
    sync_results: List[Tuple[float, int, int]] = []  # (score, start_sample, source_sec_idx)
    
    # Get sync bins and codes from first placement entry
    sync_bins = []
    sync_codes = []
    if placements and len(placements) > 0:
        first_sync = placements[0].get("sync", {})
        sync_bins = [(int(f), int(t)) for f, t in first_sync.get("bins", [])]
        sync_codes = [first_sync.get("code", [])]
        # Collect all sync codes from all seconds
        for placement in placements:
            sync = placement.get("sync", {})
            if "code" in sync:
                sync_codes.append(sync["code"])

    print(f"Stage A: Scanning {len(starts)} windows with source sync...")
    for i, st in enumerate(starts):
        win = wav[:, st:st + WIN_S]
        if win.size(-1) < WIN_S:
            win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
        
        x = win.to(args.device).unsqueeze(0)
        
        if sync_bins and sync_codes:
            # Use source sync correlation
            score, source_sec_idx = _correlate_sync_with_source(model, x, n_fft, hop, sync_bins, sync_codes)
            sync_results.append((score, st, source_sec_idx))
        else:
            # Fallback to legacy method
            sec_index = st // WIN_S
            score = _correlate_sync(model, x, n_fft=n_fft, hop=hop, sync_cfg=sync_cfg, sec_index=sec_index)
            sync_results.append((score, st, sec_index))

    # Keep top-K and cluster contiguous
    sync_results.sort(key=lambda z: z[0], reverse=True)
    top = sync_results[:max(1, args.topk)]
    # simple clustering by proximity (<= 0.75s apart)
    top_starts = sorted([(st, src_sec) for (_s, st, src_sec) in top])
    segments: List[Tuple[int, int, int]] = []  # (start, end, source_sec_offset)
    if top_starts:
        cur_s, cur_e, cur_src = top_starts[0][0], top_starts[0][0] + WIN_S, top_starts[0][1]
        for s, src_sec in top_starts[1:]:
            if s - cur_e <= int(0.75 * TARGET_SR):
                cur_e = s + WIN_S
                # Use the most common source second in the segment
                if src_sec == cur_src:
                    cur_src = src_sec
            else:
                segments.append((cur_s, cur_e, cur_src))
                cur_s, cur_e, cur_src = s, s + WIN_S, src_sec
        segments.append((cur_s, cur_e, cur_src))

    print(f"Stage B: Processing {len(segments)} candidate segments...")
    
    # Stage B: targeted NN decode around segments
    detections: List[Detection] = []
    rs_payload_bytes = int(payload_cfg.get("payload_bytes", 125))
    interleave_depth = int(payload_cfg.get("interleave_depth", 4))

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
        
        # extend by ±1s margin
        s = max(0, seg_s - WIN_S)
        e = min(wav.size(-1), seg_e + WIN_S)
        
        # Collect votes with weighted voting
        all_votes: Dict[int, List[Tuple[int, float]]] = {}  # bit_idx -> [(vote, weight), ...]
        window_count = 0
        
        cursor = s
        while cursor < e:
            win = wav[:, cursor:cursor + WIN_S]
            if win.size(-1) < WIN_S:
                win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
            
            x = win.to(args.device).unsqueeze(0)
            M_rec = model.decode(x)  # [1,2,F,T]
            
            # Calculate source second index for this window
            window_source_sec = source_sec_offset + (cursor - seg_s) // WIN_S
            pl = get_source_placements(window_source_sec)
            
            if not pl:
                # fallback: allocate slots from received audio content
                Xrx = model.stft(x)
                slots, _amp = allocate_slots_and_amplitudes(Xrx, TARGET_SR, n_fft, target_bits=167*8, amp_safety=1.0)
                pl = [(int(f), int(t), i, 1.0) for i, (f, t) in enumerate(slots[:167*8])]

            for (f, t, bit_idx, amp_weight) in pl:
                val = float(M_rec[0, 0, f, t].item())
                bit = 1 if val >= 0.0 else 0
                # Use amplitude as weight for voting
                all_votes.setdefault(bit_idx, []).append((bit, abs(amp_weight)))
            
            window_count += 1
            cursor += WIN_S

        if all_votes:
            # Weighted majority vote per bit index
            max_idx = max(all_votes.keys())
            bits: List[int] = []
            bit_weights: List[float] = []
            
            for i in range(max_idx + 1):
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

            # Calculate confidence and BER
            margins = []
            for i in range(len(bits)):
                votes = all_votes.get(i, [])
                if votes:
                    total_weight = sum(weight for _, weight in votes)
                    weighted_ones = sum(weight for bit, weight in votes if bit == 1)
                    p = weighted_ones / max(1e-6, total_weight)
                    margins.append(abs(p - 0.5) * 2.0)
            
            confidence = float(sum(margins) / max(1, len(margins))) if margins else 0.0

            # RS decode
            byte_stream = _bits_to_bytes(bits)
            deint = deinterleave_bytes(byte_stream, interleave_depth)
            rs_ok = False
            recovered_payload: Optional[bytes] = None
            
            try:
                payload = rs_decode_167_125(deint)
                recovered_payload = payload[:rs_payload_bytes]
                rs_ok = True
            except Exception as e:
                print(f"    RS decode failed: {e}")
                recovered_payload = None

            # Calculate BER
            ber = 0.0
            if recovered_payload is not None:
                # Re-encode to calculate BER
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
            if recovered_payload is not None:
                # Count windows that agree with the decoded payload
                try:
                    rs_code = rs_encode_167_125(recovered_payload)
                    interleaved = interleave_bytes(rs_code, interleave_depth)
                    expected_bits = _bytes_to_bits(interleaved)
                    
                    cursor = s
                    while cursor < e:
                        win = wav[:, cursor:cursor + WIN_S]
                        if win.size(-1) < WIN_S:
                            win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
                        
                        x = win.to(args.device).unsqueeze(0)
                        M_rec = model.decode(x)
                        
                        window_source_sec = source_sec_offset + (cursor - seg_s) // WIN_S
                        pl = get_source_placements(window_source_sec)
                        
                        if not pl:
                            Xrx = model.stft(x)
                            slots, _amp = allocate_slots_and_amplitudes(Xrx, TARGET_SR, n_fft, target_bits=167*8, amp_safety=1.0)
                            pl = [(int(f), int(t), i, 1.0) for i, (f, t) in enumerate(slots[:167*8])]

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

            # Create detection
            if recovered_payload is not None:
                try:
                    text = recovered_payload.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
            else:
                text = ""

            # Use hysteresis to expand/contract detected segment
            start_s = float(seg_s / TARGET_SR)
            end_s = float(seg_e / TARGET_SR)
            
            # Simple hysteresis: expand by 0.5s on each side if confidence is high
            if confidence > 0.7:
                start_s = max(0.0, start_s - 0.5)
                end_s = min(wav.size(-1) / TARGET_SR, end_s + 0.5)

            detection = Detection(
                start_s=start_s,
                end_s=end_s,
                confidence=confidence,
                ber=ber,
                rs_ok=rs_ok,
                payload_snippet=text[:256],
                source_sec_offset=source_sec_offset,
                num_windows=window_count,
                agreeing_windows=agreeing_windows
            )
            detections.append(detection)

    # Save detections
    detections_data = {
        "schema_version": "1.0",
        "audio_file": args.audio,
        "slots_file": args.slots,
        "total_detections": len(detections),
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
                "agreement_ratio": d.agreeing_windows / max(1, d.num_windows)
            }
            for d in detections
        ]
    }
    
    with open(args.out_detections, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2)

    # Report
    print(f"\nDetection Summary:")
    print(f"  Total candidates: {len(segments)}")
    print(f"  Successful detections: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\nDetection {i + 1}:")
        print(f"  Time: {det.start_s:.2f}s - {det.end_s:.2f}s")
        print(f"  Confidence: {det.confidence:.3f}")
        print(f"  BER: {det.ber:.3f}")
        print(f"  RS Success: {det.rs_ok}")
        print(f"  Source offset: {det.source_sec_offset}")
        print(f"  Windows: {det.agreeing_windows}/{det.num_windows} agreeing")
        if det.payload_snippet:
            print(f"  Payload: {det.payload_snippet}")
        else:
            print("  Payload: (empty)")
    
    if not detections:
        print("No payload recovered")


if __name__ == "__main__":
    main()




