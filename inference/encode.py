#!/usr/bin/env python3
"""
Encode watermark into an audio file with global slot planning and sync.

Features implemented (demo-focused):
- Payload builder: packs metadata -> RS(167,125) -> interleave -> bits
- Global slot planning: 1s windows, psychoacoustic allocation per second
- Bit repetition across track (default r=3), deterministic seeded assignment
- Sync signal per second (reserved TF bins) with Barker-13 BPSK at lower amp
- Amplitude budgeting via per-slot scaling from psychoacoustic thresholds
- Outputs: watermarked WAV and slots JSON with full specs for decoding

Defaults:
- Input: inference/Track2.wav
- Checkpoint: checkpoints/inn_decode_best.pt
- Output watermarked: watermarked.wav
- Output slots JSON: slots_map.json
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import math
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F

# Resolve paths relative to script dir and repo root
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.payload_codec import pack_fields
from pipeline.ingest_and_chunk import (
    rs_encode_167_125,
    interleave_bytes,
    allocate_slots_and_amplitudes,
)


TARGET_SR = 22050
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    if sr == TARGET_SR:
        return wav, sr
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav, TARGET_SR


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int, str]:
    # Prefer soundfile for robust WAV/AIFF reading; fallback to torchaudio
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # data: [T, C] -> mono
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        wav = torch.from_numpy(data.T)  # [1, T]
    except Exception:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
    # normalize peak to 1.0
    wav = wav / (wav.abs().max() + 1e-9)
    wav, sr = _resample_if_needed(wav, sr)
    # robust fingerprint
    file_hash_hex = _sha256_file(path)
    return wav, sr, file_hash_hex


def _chunk_audio_1s(wav: torch.Tensor) -> list[torch.Tensor]:
    T = wav.size(-1)
    chunks: list[torch.Tensor] = []
    cursor = 0
    while cursor < T:
        end = min(cursor + CHUNK_SAMPLES, T)
        ch = wav[..., cursor:end]
        if ch.size(-1) < CHUNK_SAMPLES:
            ch = F.pad(ch, (0, CHUNK_SAMPLES - ch.size(-1)))
        chunks.append(ch)
        cursor += CHUNK_SAMPLES
    if len(chunks) == 0:
        chunks = [F.pad(wav[..., :0], (0, CHUNK_SAMPLES))]
    return chunks


def _bytes_to_bits(by: bytes) -> List[int]:
    bits: List[int] = []
    for b in by:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


def _barker13_sequence() -> List[int]:
    # Barker-13 (1,1,1,1,1,0,0,1,1,0,1,0,1) -> map 1->+1, 0->-1
    seq01 = [1,1,1,1,1,0,0,1,1,0,1,0,1]
    return [1 if v == 1 else -1 for v in seq01]


def _build_sync_bins(Fbins: int, Tframes: int, sr: int, n_fft: int, K_freq: int = 8, K_time: int = 3,
                     f_low_hz: float = 200.0, f_high_hz: float = 2000.0) -> List[Tuple[int, int]]:
    # Evenly spaced frequency bins between 200–2000 Hz, and K_time evenly spaced frames per second
    f_max = sr / 2.0
    low_bin = max(1, int((f_low_hz / f_max) * (n_fft // 2)))
    high_bin = min(n_fft // 2, int((f_high_hz / f_max) * (n_fft // 2)))
    if high_bin <= low_bin:
        low_bin, high_bin = 1, max(2, n_fft // 4)
    # frequency indices
    if K_freq <= 1:
        freq_bins = [max(1, (low_bin + high_bin) // 2)]
    else:
        step = (high_bin - low_bin) / float(K_freq + 1)
        freq_bins = [int(low_bin + (i + 1) * step) for i in range(K_freq)]
    # time frames
    if K_time <= 1:
        time_frames = [max(0, (Tframes - 1) // 2)]
    else:
        step_t = max(1.0, (Tframes - 1) / float(K_time + 1))
        time_frames = [int(round((i + 1) * step_t)) for i in range(K_time)]
        time_frames = [min(tf, Tframes - 1) for tf in time_frames]
    bins: List[Tuple[int,int]] = []
    # interleave to spread across time and frequency
    for ti in time_frames:
        for fi in freq_bins:
            bins.append((fi, ti))
    return bins


@dataclass
class SyncSpec:
    code_family: str
    code_length: int
    amp: float
    k_freq: int
    k_time: int
    f_low_hz: float
    f_high_hz: float


@dataclass
class PayloadSpec:
    payload_bytes: int
    rs_n: int
    rs_k: int
    interleave_depth: int
    bit_order: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default=os.path.join(SCRIPT_DIR, "Track2.wav"))
    parser.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "checkpoints", "inn_decode_best.pt"))
    parser.add_argument("--out_wav", type=str, default=os.path.join(ROOT, "watermarked.wav"))
    parser.add_argument("--slots_json", type=str, default=os.path.join(ROOT, "slots_map.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Payload inputs
    parser.add_argument("--metadata", type=str, default="ISRC=ISRC12345678910;ISWC=ISWC12345678910;Dur=260s;RDate=2025-10-10")
    parser.add_argument("--payload_bytes", type=int, default=125, help="Raw payload bytes before RS")
    parser.add_argument("--interleave", type=int, default=4)
    # Planner and embedding
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--per_sec_capacity", type=int, default=512, help="Max payload slots per second")
    parser.add_argument("--repeat", type=int, default=3, help="Repetitions per bit across placements")
    parser.add_argument("--base_symbol_amp", type=float, default=0.08)
    parser.add_argument("--sync_rel_amp", type=float, default=0.5, help="sync amp = rel * base_symbol_amp")
    parser.add_argument("--sync_k_freq", type=int, default=8)
    parser.add_argument("--sync_k_time", type=int, default=3)
    parser.add_argument("--sync_f_low", type=float, default=200.0)
    parser.add_argument("--sync_f_high", type=float, default=2000.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    # Resolve absolute input paths (in case user passed relative paths from another CWD)
    audio_path = os.path.abspath(args.audio)
    ckpt_path = os.path.abspath(args.ckpt)
    out_wav_path = os.path.abspath(args.out_wav)
    slots_json_path = os.path.abspath(args.slots_json)

    # Basic existence checks with helpful messages
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Ensure output directories exist
    out_dir = os.path.dirname(out_wav_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    slots_dir = os.path.dirname(slots_json_path)
    if slots_dir:
        os.makedirs(slots_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    cfg = state.get("cfg", {})
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop", 512))
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(args.device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    # Load audio
    print(f"Loading audio: {audio_path}")
    wav, sr, file_hash_hex = _load_audio_mono(audio_path)
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected SR after resample: {sr}")
    chunks = _chunk_audio_1s(wav)  # list of [1,T]

    # Build payload bytes from metadata string using 6-bit packer (demo)
    # metadata format: key=value;key=value;...
    fields: Dict[str, str] = {}
    for part in args.metadata.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()
        else:
            fields["msg"] = part.strip()
    raw_bytes = pack_fields(fields)
    if len(raw_bytes) < args.payload_bytes:
        raw_bytes = raw_bytes + bytes(args.payload_bytes - len(raw_bytes))
    else:
        raw_bytes = raw_bytes[:args.payload_bytes]

    # RS encode + interleave
    rs_code = rs_encode_167_125(raw_bytes)  # 167 bytes
    rs_n, rs_k = 167, 125
    interleaved = interleave_bytes(rs_code, args.interleave)
    all_bits: List[int] = _bytes_to_bits(interleaved)

    # Plan per-second payload slots and sync bins
    placements_per_sec: List[List[Tuple[int,int,float,int]]] = []  # [(f,t,amp,bit_index)]
    sync_per_sec: List[Dict[str, object]] = []

    # Precompute sync code
    barker = _barker13_sequence()
    # expand/tiling to K_freq*K_time length (deterministic; optionally rotate per-second)

    print("Planning slots and sync per second...")
    # First pass: compute per-second candidate payload slots
    per_sec_candidates: List[List[Tuple[int,int,float]]] = []  # [(f,t,amp_scale)]
    for sec_idx, ch in enumerate(chunks):
        x = ch.to(args.device).unsqueeze(0)
        X = model.stft(x)  # [1,2,F,T]
        Fbins, Tframes = X.shape[-2], X.shape[-1]

        # Sync bins for this second
        sync_bins = _build_sync_bins(Fbins, Tframes, sr=TARGET_SR, n_fft=n_fft, K_freq=args.sync_k_freq, K_time=args.sync_k_time,
                                     f_low_hz=args.sync_f_low, f_high_hz=args.sync_f_high)
        # Construct sync code vector for these bins by tiling Barker-13 and rotating by sec_idx
        K = len(sync_bins)
        code_vec = (barker * ((K + len(barker) - 1) // len(barker)))[:K]
        # rotate by sec index to encode second index (optional indexing)
        if K > 0 and sec_idx % K != 0:
            r = sec_idx % K
            code_vec = code_vec[r:] + code_vec[:r]
        sync_per_sec.append({
            "bins": [[int(f), int(t)] for (f, t) in sync_bins],
            "code": code_vec,
        })

        # Payload candidate slots from psychoacoustic allocator
        slots, amp_per_slot = allocate_slots_and_amplitudes(
            X, TARGET_SR, n_fft, target_bits=min(args.per_sec_capacity, 167 * 8), amp_safety=1.0
        )
        # Remove any slots that collide with sync bins
        sync_set = set(sync_bins)
        cand: List[Tuple[int,int,float]] = []
        for i, (f, t) in enumerate(slots):
            if (f, t) in sync_set:
                continue
            amp_scale = float(amp_per_slot[i]) if i < len(amp_per_slot) else 1.0
            cand.append((int(f), int(t), amp_scale))
        per_sec_candidates.append(cand)

    # Flatten global placements list from candidates
    global_slots: List[Tuple[int,int,int,float]] = []  # (sec_idx, f, t, amp_scale)
    for sidx, cand in enumerate(per_sec_candidates):
        for (f, t, a) in cand:
            global_slots.append((sidx, f, t, a))

    # Deterministic shuffle using seed to avoid bias and enforce spacing
    rng = random.Random(args.seed)
    rng.shuffle(global_slots)

    # Optional simple spacing: avoid placing two payload symbols at identical (f,t) in same second twice
    used_positions: Dict[Tuple[int,int,int], bool] = {}
    filtered_global: List[Tuple[int,int,int,float]] = []
    for sidx, f, t, a in global_slots:
        key = (sidx, f, t)
        if key in used_positions:
            continue
        used_positions[key] = True
        filtered_global.append((sidx, f, t, a))

    # Assign bits with repetition r across the filtered placements
    num_bits = len(all_bits)
    need = num_bits * max(1, args.repeat)
    if len(filtered_global) < need:
        # reduce repetition if capacity is insufficient
        args.repeat = max(1, len(filtered_global) // max(1, num_bits))
        need = num_bits * args.repeat
    filtered_global = filtered_global[:need]

    # Build per-second placement lists with bit indices and amplitudes
    print(f"Total bits: {num_bits}, repetition: {args.repeat}, using placements: {len(filtered_global)}")
    placements_per_sec = [[] for _ in range(len(chunks))]
    for k, (sidx, f, t, a) in enumerate(filtered_global):
        bit_index = k % num_bits  # round-robin ensures r repeats per bit
        bit_val = all_bits[bit_index]
        amp = (1.0 if bit_val == 1 else -1.0) * args.base_symbol_amp * float(a)
        placements_per_sec[sidx].append((int(f), int(t), float(amp), int(bit_index)))

    # Encode per second, adding sync spectrogram on top of payload spectrogram
    sync_amp = args.sync_rel_amp * args.base_symbol_amp
    wm_chunks: List[torch.Tensor] = []
    print("Embedding per second...")
    for sec_idx, ch in enumerate(chunks):
        x = ch.to(args.device).unsqueeze(0)  # [1,1,T]
        X = model.stft(x)
        Fbins, Tframes = X.shape[-2], X.shape[-1]
        M_spec = torch.zeros_like(X)

        # Payload symbols
        for (f, t, amp, _bit_idx) in placements_per_sec[sec_idx]:
            if 0 <= f < Fbins and 0 <= t < Tframes:
                M_spec[0, 0, f, t] = float(amp)

        # Sync symbols (BPSK on real channel)
        sync_bins = sync_per_sec[sec_idx]["bins"]
        code_vec = sync_per_sec[sec_idx]["code"]
        for (code, ft) in zip(code_vec, sync_bins):
            f, t = int(ft[0]), int(ft[1])
            if 0 <= f < Fbins and 0 <= t < Tframes:
                M_spec[0, 0, f, t] += float(code) * float(sync_amp)

        x_wm, _ = model.encode(x, M_spec)
        x_wm = torch.clamp(x_wm, -1.0, 1.0)
        wm_chunks.append(x_wm)

    # Concatenate chunks and save
    x_wm_full = torch.cat([ch.squeeze(0) for ch in wm_chunks], dim=-1)  # [1,T]
    print(f"Saving watermarked audio to: {out_wav_path}")
    torchaudio.save(out_wav_path, x_wm_full.detach().cpu(), sample_rate=TARGET_SR)

    # Build JSON slots map
    fingerprint = file_hash_hex
    slots_json = {
        "audio_fingerprint": fingerprint,
        "sample_rate": TARGET_SR,
        "stft": {"n_fft": n_fft, "hop": hop, "win_length": n_fft},
        "planner_seed": int(args.seed),
        "sync_spec": asdict(SyncSpec(
            code_family="barker13",
            code_length=len(sync_per_sec[0]["code"]) if len(sync_per_sec) > 0 else 0,
            amp=float(sync_amp),
            k_freq=int(args.sync_k_freq),
            k_time=int(args.sync_k_time),
            f_low_hz=float(args.sync_f_low),
            f_high_hz=float(args.sync_f_high),
        )),
        "payload_spec": asdict(PayloadSpec(
            payload_bytes=int(args.payload_bytes),
            rs_n=167,
            rs_k=125,
            interleave_depth=int(args.interleave),
            bit_order="LSB-first-per-byte",
        )),
        "repeat": int(args.repeat),
        "base_symbol_amp": float(args.base_symbol_amp),
        "placements": [],  # list per second
    }

    for sec_idx in range(len(chunks)):
        sec_entry = {
            "second_index": sec_idx,
            "payload": [
                [int(f), int(t), float(amp), int(bit_idx)]
                for (f, t, amp, bit_idx) in placements_per_sec[sec_idx]
            ],
            "sync": sync_per_sec[sec_idx],
        }
        slots_json["placements"].append(sec_entry)

    print(f"Writing slots map to: {slots_json_path}")
    with open(slots_json_path, "w", encoding="utf-8") as f:
        json.dump(slots_json, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()


