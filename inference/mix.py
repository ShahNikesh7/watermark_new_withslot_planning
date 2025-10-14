#!/usr/bin/env python3
"""
Mix a 3s watermarked clip into a host track with loudness match and random placement.

Inputs:
- Watermarked master (default: watermarked.wav)
- Host track (default: inference/sample.mp3)

Process:
- Select a 3s clip from watermarked master with healthy energy (avoid silence); apply 20 ms fade-in/out
- Loudness-match to host segment (RMS proxy); optionally overlay at -3 dB or hard-replace
- Choose random insertion point away from first/last 10% of host
- Save output sampled.wav and sidecar JSON with ground-truth timestamps
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from typing import Tuple

import torch
import torchaudio
import torch.nn.functional as F


TARGET_SR = 22050
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    if sr == TARGET_SR:
        return wav, sr
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav, TARGET_SR


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # [C,T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav, sr = _resample_if_needed(wav, sr)
    # peak-normalize to avoid clipping interactions
    peak = wav.abs().max().item()
    if peak > 0:
        wav = wav / peak
    return wav, sr


def _energy_per_window(wav: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    # Simple RMS energy per frame
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    x = wav.unsqueeze(0).unsqueeze(0)  # [1,1,T]
    w = torch.ones(1, 1, win, device=wav.device) / float(win)
    mean_sq = F.conv1d(x * x, w, stride=hop, padding=0)  # [1,1,F]
    return mean_sq[0,0]


def _find_best_clip(wm: torch.Tensor) -> int:
    # Return start sample index for a 3s window maximizing energy, ignoring first/last 0.5s of wm if possible
    T = wm.size(-1)
    if T <= CLIP_SAMPLES:
        return 0
    hop = int(0.25 * TARGET_SR)
    win = CLIP_SAMPLES
    energies = _energy_per_window(wm, win=win, hop=hop)
    best_idx = int(torch.argmax(energies).item())
    start = best_idx * hop
    start = max(0, min(start, T - CLIP_SAMPLES))
    return start


def _apply_fade(x: torch.Tensor, fade_ms: float = 20.0) -> torch.Tensor:
    # x: [1,T]
    T = x.size(-1)
    fade_len = int((fade_ms / 1000.0) * TARGET_SR)
    if fade_len <= 0 or fade_len * 2 >= T:
        return x
    ramp_in = torch.linspace(0.0, 1.0, fade_len, device=x.device)
    ramp_out = torch.linspace(1.0, 0.0, fade_len, device=x.device)
    x[:, :fade_len] *= ramp_in
    x[:, -fade_len:] *= ramp_out
    return x


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x * x).mean() + 1e-12).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", type=str, default="watermarked.wav")
    parser.add_argument("--host", type=str, default=os.path.join("inference", "sample.mp3"))
    parser.add_argument("--out", type=str, default="sampled.wav")
    parser.add_argument("--sidecar", type=str, default="sampled_meta.json")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--blend_db", type=float, default=0.0, help="<=0 for overlay attenuation; 0 means hard-replace")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load audio
    wm, sr_wm = _load_audio_mono(args.wm)
    host, sr_h = _load_audio_mono(args.host)
    if sr_wm != TARGET_SR or sr_h != TARGET_SR:
        raise RuntimeError("Unexpected SR after resample")

    # Pick a 3s high-energy clip from watermarked
    start_wm = _find_best_clip(wm)
    wm_clip = wm[:, start_wm:start_wm + CLIP_SAMPLES]
    if wm_clip.size(-1) < CLIP_SAMPLES:
        wm_clip = F.pad(wm_clip, (0, CLIP_SAMPLES - wm_clip.size(-1)))
    wm_clip = _apply_fade(wm_clip, fade_ms=20.0)

    # Choose random insertion point away from first/last 10%
    T_host = host.size(-1)
    margin = int(0.1 * T_host)
    if T_host <= (2 * margin + CLIP_SAMPLES):
        insert = max(0, (T_host - CLIP_SAMPLES) // 2)
    else:
        lo = margin
        hi = T_host - margin - CLIP_SAMPLES
        insert = random.randint(lo, hi)

    # Loudness match (RMS) to host segment
    host_seg = host[:, insert:insert + CLIP_SAMPLES]
    if host_seg.size(-1) < CLIP_SAMPLES:
        host_seg = F.pad(host_seg, (0, CLIP_SAMPLES - host_seg.size(-1)))
    rms_host = _rms(host_seg)
    rms_wm = _rms(wm_clip)
    if rms_wm > 0:
        scale = rms_host / rms_wm
    else:
        scale = 1.0
    wm_clip = wm_clip * float(scale)

    # Overlay or hard-replace
    out = host.clone()
    if args.blend_db < -1e-6:
        # overlay at attenuation
        gain = 10.0 ** (args.blend_db / 20.0)
        out[:, insert:insert + CLIP_SAMPLES] = torch.clamp(
            host_seg + wm_clip * gain, -1.0, 1.0
        )
        mode = "overlay"
    else:
        # hard replace
        out[:, insert:insert + CLIP_SAMPLES] = torch.clamp(wm_clip, -1.0, 1.0)
        mode = "replace"

    # Save output and sidecar
    torchaudio.save(args.out, out.detach().cpu(), sample_rate=TARGET_SR)
    meta = {
        "mode": mode,
        "sample_rate": TARGET_SR,
        "clip_seconds": CLIP_SECONDS,
        "insert_start_samples": int(insert),
        "insert_end_samples": int(insert + CLIP_SAMPLES),
        "insert_start_sec": float(insert / TARGET_SR),
        "insert_end_sec": float((insert + CLIP_SAMPLES) / TARGET_SR),
        "wm_source_start_samples": int(start_wm),
        "wm_source_start_sec": float(start_wm / TARGET_SR),
    }
    with open(args.sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {args.out}\nSidecar: {args.sidecar}")


if __name__ == "__main__":
    main()




