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
from typing import Dict, List, Tuple

import torch
import torchaudio

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.ingest_and_chunk import (
    rs_decode_167_125,
    deinterleave_bytes,
    allocate_slots_and_amplitudes,
)


TARGET_SR = 22050
SEC = 1.0
WIN_S = int(TARGET_SR * SEC)


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


def _correlate_sync(model: INNWatermarker, x: torch.Tensor, n_fft: int, hop: int, sync_cfg: Dict, sec_index: int) -> float:
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
    # Use real part for BPSK readout
    vals = []
    for (f, t) in bins:
        vals.append(float(X[0, 0, f, t].item()))
    if len(vals) == 0:
        return 0.0
    # normalized correlation
    v = torch.tensor(vals, dtype=torch.float32)
    c = torch.tensor(code, dtype=torch.float32)
    v = (v - v.mean()) / (v.std() + 1e-6)
    c = (c - c.mean()) / (c.std() + 1e-6)
    score = float(torch.dot(v, c) / max(1, len(vals)))
    return score


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
    args = parser.parse_args()

    # Load slots config
    with open(args.slots, "r", encoding="utf-8") as f:
        slots_map = json.load(f)
    stft_cfg = slots_map.get("stft", {"n_fft": 1024, "hop": 512, "win_length": 1024})
    n_fft = int(stft_cfg.get("n_fft", 1024))
    hop = int(stft_cfg.get("hop", 512))
    sync_cfg = slots_map.get("sync_spec", {})
    payload_cfg = slots_map.get("payload_spec", {})
    repeat = int(slots_map.get("repeat", 1))
    placements = slots_map.get("placements", [])

    # Build model
    state = torch.load(args.ckpt, map_location=args.device)
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(args.device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    # Load audio
    wav, sr = _load_audio_mono(args.audio)
    if sr != TARGET_SR:
        raise RuntimeError("Unexpected SR after resample")

    # Stage A: sync scan
    hop_samp = max(1, int(args.hop_sec * TARGET_SR))
    starts = _slide_indices(wav.size(-1), hop=hop_samp)
    scores: List[Tuple[float, int]] = []
    for i, st in enumerate(starts):
        win = wav[:, st:st + WIN_S]
        if win.size(-1) < WIN_S:
            win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
        sec_index = st // WIN_S
        x = win.to(args.device).unsqueeze(0)
        score = _correlate_sync(model, x, n_fft=n_fft, hop=hop, sync_cfg=sync_cfg, sec_index=sec_index)
        scores.append((score, st))

    # Keep top-K and cluster contiguous
    scores.sort(key=lambda z: z[0], reverse=True)
    top = scores[:max(1, args.topk)]
    # simple clustering by proximity (<= 0.75s apart)
    top_starts = sorted([st for (_s, st) in top])
    segments: List[Tuple[int, int]] = []
    if top_starts:
        cur_s, cur_e = top_starts[0], top_starts[0] + WIN_S
        for s in top_starts[1:]:
            if s - cur_e <= int(0.75 * TARGET_SR):
                cur_e = s + WIN_S
            else:
                segments.append((cur_s, cur_e))
                cur_s, cur_e = s, s + WIN_S
        segments.append((cur_s, cur_e))

    # Stage B: targeted NN decode around segments
    recovered_payload: bytes | None = None
    confidence: float = 0.0
    rs_payload_bytes = int(payload_cfg.get("payload_bytes", 125))
    interleave_depth = int(payload_cfg.get("interleave_depth", 4))

    # Helper: read placements for a given absolute second index if available
    def sec_placements(sec_idx: int) -> List[Tuple[int,int,int]]:
        if 0 <= sec_idx < len(placements):
            ent = placements[sec_idx]
            arr = ent.get("payload", [])
            out = []
            for it in arr:
                if isinstance(it, list) and len(it) >= 4:
                    f, t, amp, bit_idx = int(it[0]), int(it[1]), float(it[2]), int(it[3])
                    out.append((f, t, bit_idx))
            return out
        return []

    all_votes: Dict[int, List[int]] = {}
    for (seg_s, seg_e) in segments:
        # extend by ±1s margin
        s = max(0, seg_s - WIN_S)
        e = min(wav.size(-1), seg_e + WIN_S)
        cursor = s
        while cursor < e:
            win = wav[:, cursor:cursor + WIN_S]
            if win.size(-1) < WIN_S:
                win = torch.nn.functional.pad(win, (0, WIN_S - win.size(-1)))
            x = win.to(args.device).unsqueeze(0)
            M_rec = model.decode(x)  # [1,2,F,T]

            # determine second index to fetch placements
            sec_idx = cursor // WIN_S
            pl = sec_placements(sec_idx)
            if not pl:
                # fallback: allocate slots from received audio content
                Xrx = model.stft(x)
                slots, _amp = allocate_slots_and_amplitudes(Xrx, TARGET_SR, n_fft, target_bits=167*8, amp_safety=1.0)
                pl = [(int(f), int(t), i) for i, (f, t) in enumerate(slots[:167*8])]

            for (f, t, bit_idx) in pl:
                val = float(M_rec[0, 0, f, t].item())
                bit = 1 if val >= 0.0 else 0
                all_votes.setdefault(bit_idx, []).append(bit)

            cursor += WIN_S

    if all_votes:
        # majority vote per bit index, then flatten by bit index order
        max_idx = max(all_votes.keys())
        bits: List[int] = []
        for i in range(max_idx + 1):
            votes = all_votes.get(i, [])
            if votes:
                one = sum(votes)
                zero = len(votes) - one
                bit = 1 if one >= zero else 0
                bits.append(bit)
        # Confidence as mean margin
        margins = []
        for i in range(len(bits)):
            votes = all_votes.get(i, [])
            if votes:
                p = sum(votes) / float(len(votes))
                margins.append(abs(p - 0.5) * 2.0)
        confidence = float(sum(margins) / max(1, len(margins))) if margins else 0.0

        byte_stream = _bits_to_bytes(bits)
        deint = deinterleave_bytes(byte_stream, interleave_depth)
        try:
            payload = rs_decode_167_125(deint)
            recovered_payload = payload[:rs_payload_bytes]
        except Exception:
            recovered_payload = None

    if recovered_payload is not None:
        try:
            text = recovered_payload.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = ""

    # Report
    print(f"Candidates: {len(segments)}; Confidence: {confidence:.3f}")
    if text:
        print(f"Recovered metadata: {text[:256]}")
    else:
        print("No payload recovered")


if __name__ == "__main__":
    main()




