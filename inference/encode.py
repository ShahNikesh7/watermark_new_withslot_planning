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
import gc
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F

# Memory optimization settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

# Memory optimization functions
def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def process_chunk_memory_efficient(model, chunk, device, clear_cache=True):
    """Process a single chunk with memory optimization and transient preservation"""
    if clear_cache:
        clear_gpu_memory()
    
    # Move chunk to device
    chunk = chunk.to(device, non_blocking=True)
    
    # Process with gradient disabled to save memory
    with torch.no_grad():
        # Use higher precision for better transient preservation
        result = model.stft(chunk.unsqueeze(0))
        # Preserve numerical precision
        result = result.float()
    
    # Move result back to CPU immediately to free GPU memory
    result_cpu = result.cpu()
    del chunk, result
    if clear_cache:
        clear_gpu_memory()
    
    return result_cpu

def encode_chunk_memory_efficient(model, chunk, m_spec, device, clear_cache=True):
    """Encode a single chunk with memory optimization and transient preservation"""
    if clear_cache:
        clear_gpu_memory()
    
    # Move tensors to device
    chunk = chunk.to(device, non_blocking=True)
    m_spec = m_spec.to(device, non_blocking=True)
    
    # Process with gradient disabled and autocast for memory efficiency
    with torch.no_grad():
        # Use autocast for mixed precision if available
        if device == 'cuda':
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
                x_wm, _ = model.encode(chunk, m_spec)
                # Preserve numerical precision and avoid over-clamping
                x_wm = x_wm.float()
                # More conservative clamping to preserve dynamics
                x_wm = torch.clamp(x_wm, -0.99, 0.99)
        else:
            x_wm, _ = model.encode(chunk, m_spec)
            # Preserve numerical precision and avoid over-clamping
            x_wm = x_wm.float()
            # More conservative clamping to preserve dynamics
            x_wm = torch.clamp(x_wm, -0.99, 0.99)
    
    # Move result back to CPU immediately
    result_cpu = x_wm.cpu()
    del chunk, m_spec, x_wm
    if clear_cache:
        clear_gpu_memory()
    
    return result_cpu


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


def _load_audio_stereo(path: str) -> tuple[torch.Tensor, int, str]:
    """Load audio preserving original characteristics - no normalization"""
    # Prefer soundfile for robust WAV/AIFF reading; fallback to torchaudio
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # data: [T, C] -> [C, T] for torch
        wav = torch.from_numpy(data.T)  # [C, T]
    except Exception:
        wav, sr = torchaudio.load(path)
    
    # robust fingerprint
    file_hash_hex = _sha256_file(path)
    return wav, sr, file_hash_hex


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int, str]:
    """Load audio as mono for processing - no normalization"""
    wav, sr, file_hash_hex = _load_audio_stereo(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr, file_hash_hex


def _measure_lufs(wav: torch.Tensor, sr: int) -> float:
    """Measure integrated LUFS using improved RMS approximation with better dynamics preservation"""
    # Use windowed RMS to better capture dynamics and avoid over-smoothing
    window_size = int(sr * 0.4)  # 400ms windows for better transient capture
    if wav.size(-1) < window_size:
        window_size = wav.size(-1)
    
    # Calculate RMS in overlapping windows to preserve transients
    rms_values = []
    for i in range(0, wav.size(-1) - window_size + 1, window_size // 2):
        window = wav[..., i:i + window_size]
        rms = torch.sqrt(torch.mean(window * window))
        rms_values.append(rms)
    
    if not rms_values:
        rms = torch.sqrt(torch.mean(wav * wav))
    else:
        # Use 90th percentile RMS to avoid being dominated by quiet sections
        rms_tensor = torch.stack(rms_values)
        rms = torch.quantile(rms_tensor, 0.9)
    
    lufs = 20.0 * torch.log10(rms + 1e-9) - 0.691
    return float(lufs)


def _preserve_dynamic_range(original: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
    """Preserve the original dynamic range to prevent volume ducking"""
    # Calculate original dynamic range metrics
    orig_peak = original.abs().max()
    orig_rms = torch.sqrt(torch.mean(original * original))
    
    # Calculate watermarked dynamic range metrics
    wm_peak = watermarked.abs().max()
    wm_rms = torch.sqrt(torch.mean(watermarked * watermarked))
    
    # Calculate scaling factors to preserve dynamic range
    peak_ratio = orig_peak / (wm_peak + 1e-9)
    rms_ratio = orig_rms / (wm_rms + 1e-9)
    
    # Use the more conservative scaling to avoid over-amplification
    scale_factor = min(peak_ratio, rms_ratio, 1.0)  # Don't amplify beyond original
    
    return watermarked * scale_factor

def _apply_loudness_matching(original: torch.Tensor, watermarked: torch.Tensor, target_lufs: float) -> torch.Tensor:
    """Apply loudness matching to preserve original LUFS while maintaining dynamics"""
    # First preserve dynamic range to prevent volume ducking
    watermarked = _preserve_dynamic_range(original, watermarked)
    
    current_lufs = _measure_lufs(watermarked, TARGET_SR)
    lufs_diff = target_lufs - current_lufs
    gain_db = lufs_diff
    gain_linear = 10.0 ** (gain_db / 20.0)
    
    # Apply gain but preserve true-peak headroom and dynamics
    matched = watermarked * gain_linear
    
    # More conservative limiting to preserve transients
    peak = matched.abs().max()
    if peak > 0.98:  # Only limit if very close to clipping
        # Use gentle compression instead of hard limiting to preserve transients
        ratio = 0.98 / peak
        # Apply gentle compression curve to preserve dynamics
        matched = torch.tanh(matched * (1.0 / 0.98)) * 0.98
    
    return matched


def _extract_mid_side(stereo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mid (sum) and side (difference) components from stereo"""
    if stereo.size(0) != 2:
        # If not stereo, return as-is for mid, zeros for side
        return stereo, torch.zeros_like(stereo)
    
    mid = (stereo[0] + stereo[1]) / 2.0  # (L + R) / 2
    side = (stereo[0] - stereo[1]) / 2.0  # (L - R) / 2
    return mid.unsqueeze(0), side.unsqueeze(0)


def _reconstruct_stereo(mid: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Reconstruct stereo from mid and side components"""
    left = mid + side   # L = M + S
    right = mid - side  # R = M - S
    return torch.stack([left.squeeze(0), right.squeeze(0)], dim=0)


def _upsample_watermark_delta(delta_mono: torch.Tensor, target_sr: int) -> torch.Tensor:
    """Upsample watermark delta from 22.05kHz to target sample rate"""
    if target_sr == TARGET_SR:
        return delta_mono
    
    # Use high-quality resampling
    resampler = torchaudio.transforms.Resample(
        orig_freq=TARGET_SR, 
        new_freq=target_sr,
        resampling_method="sinc_interp_hann"
    )
    return resampler(delta_mono)


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


def _encode_second_index_to_sync(sec_idx: int, base_code: List[int], max_sec_idx: int = 1023) -> List[int]:
    """
    Encode the source second index into the sync code using a simple modulation scheme.
    This allows the decoder to recover the source second index from the sync correlation.
    
    Args:
        sec_idx: Source second index (0-based)
        base_code: Base Barker sequence
        max_sec_idx: Maximum expected second index (for modulo operation)
    
    Returns:
        Modified sync code that encodes the second index
    """
    if sec_idx < 0 or sec_idx > max_sec_idx:
        sec_idx = sec_idx % (max_sec_idx + 1)
    
    # Use a simple phase shift based on second index
    # This creates a unique signature for each second that can be detected
    phase_shift = sec_idx % len(base_code)
    if phase_shift == 0:
        return base_code.copy()
    
    # Rotate the code by the phase shift
    rotated_code = base_code[phase_shift:] + base_code[:phase_shift]
    
    # Add a subtle amplitude modulation to encode additional bits
    # This helps distinguish between different seconds even with similar phase shifts
    amp_mod = 1.0 + 0.1 * (sec_idx % 4)  # 4 different amplitude levels
    return [int(x * amp_mod) for x in rotated_code]


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
    parser.add_argument("--metadata", type=str, default="ISRCISRC12345678910ISWCISWC12345678910Dur260sRDate20251010")
    parser.add_argument("--payload_bytes", type=int, default=125, help="Raw payload bytes before RS")
    parser.add_argument("--interleave", type=int, default=4)
    # Planner and embedding
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--per_sec_capacity", type=int, default=512, help="Max payload slots per second")
    parser.add_argument("--repeat", type=int, default=3, help="Repetitions per bit across placements")
    parser.add_argument("--base_symbol_amp", type=float, default=0.06, help="Reduced default amplitude to prevent volume ducking")
    parser.add_argument("--sync_rel_amp", type=float, default=0.4, help="sync amp = rel * base_symbol_amp (reduced to prevent interference)")
    parser.add_argument("--sync_k_freq", type=int, default=8)
    parser.add_argument("--sync_k_time", type=int, default=3)
    parser.add_argument("--sync_f_low", type=float, default=200.0)
    parser.add_argument("--sync_f_high", type=float, default=2000.0)
    # Memory optimization options
    parser.add_argument("--batch_size", type=int, default=1, help="Process multiple seconds at once (1 for memory efficiency)")
    parser.add_argument("--clear_cache_freq", type=int, default=1, help="Clear GPU cache every N seconds")
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
    # Load checkpoint with memory optimization
    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = state.get("cfg", {})
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop", 512))
    
    # Create model and move to device
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft})
    model.load_state_dict(state.get("model_state", state), strict=False)
    model = model.to(args.device)
    model.eval()
    
    # Get model hash for reproducibility before deleting state
    model_hash = hashlib.sha256(str(state.get("model_state", {})).encode()).hexdigest()[:16]
    
    # Clear checkpoint from memory
    del state
    clear_gpu_memory()
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Load audio preserving original characteristics
    print(f"Loading audio: {audio_path}")
    wav_stereo, original_sr, file_hash_hex = _load_audio_stereo(audio_path)
    print(f"Original: {wav_stereo.shape}, {original_sr} Hz")
    
    # Extract mid and side components for stereo processing
    mid_component, side_component = _extract_mid_side(wav_stereo)
    print(f"Mid component: {mid_component.shape}, Side component: {side_component.shape}")
    
    # Measure original LUFS for loudness matching
    original_lufs = _measure_lufs(mid_component, original_sr)
    print(f"Original LUFS: {original_lufs:.2f} dB")
    
    # Resample mid component to 22.05kHz for watermarking
    if original_sr != TARGET_SR:
        print(f"Resampling mid component from {original_sr} Hz to {TARGET_SR} Hz for watermarking")
        mid_resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr, 
            new_freq=TARGET_SR,
            resampling_method="sinc_interp_hann"
        )
        mid_22k = mid_resampler(mid_component)
    else:
        mid_22k = mid_component
    
    chunks = _chunk_audio_1s(mid_22k)  # list of [1,T] at 22.05kHz

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
    
    # Compute payload checksum for verification
    payload_checksum = hashlib.sha256(interleaved).hexdigest()

    # Plan per-second payload slots and sync bins
    placements_per_sec: List[List[Tuple[int,int,float,int]]] = []  # [(f,t,amp,bit_index)]
    sync_per_sec: List[Dict[str, object]] = []

    # Precompute sync code
    barker = _barker13_sequence()
    # expand/tiling to K_freq*K_time length (deterministic; optionally rotate per-second)

    print("Planning slots and sync per second...")
    # First pass: compute per-second candidate payload slots with memory optimization
    per_sec_candidates: List[List[Tuple[int,int,float]]] = []  # [(f,t,amp_scale)]
    for sec_idx, ch in enumerate(chunks):
        print(f"Processing second {sec_idx + 1}/{len(chunks)}...")
        
        # Use memory-efficient processing
        X = process_chunk_memory_efficient(model, ch, args.device)
        Fbins, Tframes = X.shape[-2], X.shape[-1]

        # Sync bins for this second
        sync_bins = _build_sync_bins(Fbins, Tframes, sr=TARGET_SR, n_fft=n_fft, K_freq=args.sync_k_freq, K_time=args.sync_k_time,
                                     f_low_hz=args.sync_f_low, f_high_hz=args.sync_f_high)
        # Construct sync code vector for these bins by tiling Barker-13 and encoding sec_idx
        K = len(sync_bins)
        base_code = (barker * ((K + len(barker) - 1) // len(barker)))[:K]
        # Encode the source second index into the sync code
        code_vec = _encode_second_index_to_sync(sec_idx, base_code, max_sec_idx=len(chunks)-1)
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
        
        # Clear memory after each second
        del X
        if sec_idx % args.clear_cache_freq == 0:
            clear_gpu_memory()
            if torch.cuda.is_available():
                print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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
    
    # Capacity validation with clear warnings
    total_capacity = len(filtered_global)
    required_capacity = need
    if total_capacity < required_capacity:
        print(f"WARNING: Insufficient capacity for chosen repeat!")
        print(f"  Track length: {len(chunks)} seconds")
        print(f"  Per-second capacity: {args.per_sec_capacity}")
        print(f"  Total available slots: {total_capacity}")
        print(f"  Required slots (repeat={args.repeat}): {required_capacity}")
        print(f"  Reducing repeat from {args.repeat} to {max(1, total_capacity // max(1, num_bits))}")
        args.repeat = max(1, total_capacity // max(1, num_bits))
        need = num_bits * args.repeat
        if need > total_capacity:
            print(f"ERROR: Even with reduced repeat, insufficient capacity!")
            print(f"  This will result in decode confidence issues.")
            raise RuntimeError(f"Insufficient capacity: need {need}, have {total_capacity}")
    
    filtered_global = filtered_global[:need]

    # Build per-second placement lists with bit indices and amplitudes
    print(f"Total bits: {num_bits}, repetition: {args.repeat}, using placements: {len(filtered_global)}")
    print(f"First 10 bits: {all_bits[:10]}")
    placements_per_sec = [[] for _ in range(len(chunks))]
    
    # Track repetition distribution for each bit
    bit_distribution: Dict[int, Dict[str, any]] = {}
    for bit_idx in range(num_bits):
        bit_distribution[bit_idx] = {
            "seconds_covered": set(),
            "frames_covered": set(),
            "total_placements": 0
        }
    
    for k, (sidx, f, t, a) in enumerate(filtered_global):
        bit_index = k % num_bits  # round-robin ensures r repeats per bit
        bit_val = all_bits[bit_index]
        amp = (1.0 if bit_val == 1 else -1.0) * args.base_symbol_amp * float(a)
        placements_per_sec[sidx].append((int(f), int(t), float(amp), int(bit_index)))
        
        # Track distribution for this bit
        bit_distribution[bit_index]["seconds_covered"].add(sidx)
        bit_distribution[bit_index]["frames_covered"].add(t)
        bit_distribution[bit_index]["total_placements"] += 1
        
        if k < 10:  # Debug first 10 placements
            print(f"Placement {k}: sec={sidx}, bit_idx={bit_index}, bit_val={bit_val}, amp={amp:.3f}")
    
    # Convert sets to lists for JSON serialization
    for bit_idx in bit_distribution:
        bit_distribution[bit_idx]["seconds_covered"] = sorted(list(bit_distribution[bit_idx]["seconds_covered"]))
        bit_distribution[bit_idx]["frames_covered"] = sorted(list(bit_distribution[bit_idx]["frames_covered"]))

    # Encode per second, adding sync spectrogram on top of payload spectrogram
    sync_amp = args.sync_rel_amp * args.base_symbol_amp
    wm_chunks: List[torch.Tensor] = []
    clipping_telemetry: List[Dict[str, float]] = []
    print("Embedding per second...")
    for sec_idx, ch in enumerate(chunks):
        print(f"Encoding second {sec_idx + 1}/{len(chunks)}...")
        
        # Process chunk to get spectrogram dimensions
        X = process_chunk_memory_efficient(model, ch, args.device)
        Fbins, Tframes = X.shape[-2], X.shape[-1]
        
        # Create message spectrogram on CPU to save GPU memory
        M_spec = torch.zeros_like(X)

        # Payload symbols with improved amplitude scaling
        for (f, t, amp, _bit_idx) in placements_per_sec[sec_idx]:
            if 0 <= f < Fbins and 0 <= t < Tframes:
                # Scale amplitude more conservatively to preserve dynamics
                scaled_amp = float(amp) * 0.9  # Reduce watermark amplitude slightly
                M_spec[0, 0, f, t] = scaled_amp

        # Sync symbols (BPSK on imaginary channel) - use separate channel to avoid interference
        sync_bins = sync_per_sec[sec_idx]["bins"]
        code_vec = sync_per_sec[sec_idx]["code"]
        for (code, ft) in zip(code_vec, sync_bins):
            f, t = int(ft[0]), int(ft[1])
            if 0 <= f < Fbins and 0 <= t < Tframes:
                # Use imaginary channel for sync to avoid interference with payload
                # Scale sync amplitude more conservatively
                scaled_sync_amp = float(sync_amp) * 0.8  # Reduce sync amplitude
                M_spec[0, 1, f, t] = float(code) * scaled_sync_amp

        # Encode with memory optimization
        x_wm = encode_chunk_memory_efficient(model, ch, M_spec, args.device)
        wm_chunks.append(x_wm)
        
        # Record clipping telemetry for this second
        pre_clip_max = float(x_wm.abs().max().item())
        clipped_samples = int((x_wm.abs() >= 0.99).sum().item())
        total_samples = x_wm.numel()
        clipping_ratio = clipped_samples / max(1, total_samples)
        headroom_db = 20.0 * math.log10(max(1e-6, 1.0 / pre_clip_max))
        
        # Calculate average per-slot amplitude scale for this second
        slot_amps = [abs(amp) for (_, _, amp, _) in placements_per_sec[sec_idx]]
        avg_slot_amp_scale = sum(slot_amps) / max(1, len(slot_amps)) if slot_amps else 0.0
        
        clipping_telemetry.append({
            "second": sec_idx,
            "pre_clip_peak": pre_clip_max,
            "clipped_samples": clipped_samples,
            "total_samples": total_samples,
            "clipping_ratio": clipping_ratio,
            "headroom_db": headroom_db,
            "avg_slot_amp_scale": avg_slot_amp_scale
        })
        
        # Clear memory after each encoding
        del X, M_spec
        if sec_idx % args.clear_cache_freq == 0:
            clear_gpu_memory()
            if torch.cuda.is_available():
                print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Concatenate watermarked mid component
    mid_wm_22k = torch.cat([ch.squeeze(0) for ch in wm_chunks], dim=-1)  # [1,T] at 22.05kHz
    print(f"Watermarked mid component: {mid_wm_22k.shape}")
    
    # Calculate watermark delta (difference between original and watermarked)
    # Use the same chunking as the watermarked version to ensure matching lengths
    mid_original_22k = torch.cat([ch.squeeze(0) for ch in chunks], dim=-1)
    print(f"Original mid component: {mid_original_22k.shape}")
    
    # Ensure both tensors have the same length
    min_length = min(mid_wm_22k.size(-1), mid_original_22k.size(-1))
    mid_wm_22k = mid_wm_22k[..., :min_length]
    mid_original_22k = mid_original_22k[..., :min_length]
    
    watermark_delta_22k = mid_wm_22k - mid_original_22k
    print(f"Watermark delta: {watermark_delta_22k.shape}")
    
    # Upsample watermark delta to original sample rate
    if original_sr != TARGET_SR:
        print(f"Upsampling watermark delta from {TARGET_SR} Hz to {original_sr} Hz")
        watermark_delta_original = _upsample_watermark_delta(watermark_delta_22k, original_sr)
    else:
        watermark_delta_original = watermark_delta_22k
    
    # Ensure watermark delta matches original mid component length
    target_length = mid_component.size(-1)
    if watermark_delta_original.size(-1) != target_length:
        print(f"Adjusting watermark delta length from {watermark_delta_original.size(-1)} to {target_length}")
        if watermark_delta_original.size(-1) > target_length:
            # Truncate if too long
            watermark_delta_original = watermark_delta_original[..., :target_length]
        else:
            # Pad with zeros if too short
            pad_length = target_length - watermark_delta_original.size(-1)
            watermark_delta_original = F.pad(watermark_delta_original, (0, pad_length))
    
    # Apply watermark delta to original mid component with better precision
    mid_wm_original = mid_component + watermark_delta_original
    
    # Ensure we maintain the same data type for consistency
    mid_wm_original = mid_wm_original.float()
    
    # Apply loudness matching to preserve original LUFS
    print("Applying loudness matching...")
    mid_wm_matched = _apply_loudness_matching(mid_component, mid_wm_original, original_lufs)
    
    # Reconstruct stereo: watermarked mid + original side
    wav_wm_stereo = _reconstruct_stereo(mid_wm_matched, side_component)
    
    # Ensure consistent data type
    wav_wm_stereo = wav_wm_stereo.float()
    
    # Final gentle limiting to prevent clipping while preserving dynamics
    peak = wav_wm_stereo.abs().max()
    if peak > 0.99:
        print(f"Applying gentle limiting: peak was {peak:.3f}")
        # Use gentle compression instead of hard limiting to preserve transients
        wav_wm_stereo = torch.tanh(wav_wm_stereo * (1.0 / 0.99)) * 0.99
    
    # Measure final LUFS
    final_lufs = _measure_lufs(mid_wm_matched, original_sr)
    print(f"Final LUFS: {final_lufs:.2f} dB (target: {original_lufs:.2f} dB)")
    print(f"LUFS difference: {final_lufs - original_lufs:.2f} dB")
    
    print(f"Saving watermarked audio to: {out_wav_path}")
    torchaudio.save(out_wav_path, wav_wm_stereo.detach().cpu(), sample_rate=original_sr)

    # Build JSON slots map
    fingerprint = file_hash_hex
    slots_json = {
        "schema_version": "1.0",
        "audio_fingerprint": fingerprint,
        "model_id": os.path.basename(ckpt_path),
        "model_hash": model_hash,
        "original_sample_rate": original_sr,
        "processing_sample_rate": TARGET_SR,
        "stft": {"n_fft": n_fft, "hop": hop, "win_length": n_fft},
        "audio_processing": {
            "preserve_mix_master": True,
            "stereo_processing": "mid_side",
            "mid_component_watermarked": True,
            "side_component_preserved": True,
            "original_lufs": original_lufs,
            "final_lufs": final_lufs,
            "lufs_difference": final_lufs - original_lufs,
            "upsampling_method": "sinc_interp_hann",
            "loudness_matching_applied": True
        },
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
        "sync_indexing_contract": {
            "encoding_method": "phase_shift_plus_amplitude_modulation",
            "phase_shift": "sec_idx % code_length",
            "amplitude_levels": 4,
            "amplitude_modulation": "1.0 + 0.1 * (sec_idx % 4)",
            "detection_channel": "imaginary",
            "correlation_method": "normalized_dot_product",
            "recovery_rule": "test_phase_shifts_0_to_min(100, total_seconds) and select_best_correlation"
        },
        "channel_assignments": {
            "payload_channel": "real",
            "sync_channel": "imaginary",
            "rationale": "payload_on_real_avoids_interference_with_sync_on_imaginary"
        },
        "payload_spec": asdict(PayloadSpec(
            payload_bytes=int(args.payload_bytes),
            rs_n=167,
            rs_k=125,
            interleave_depth=int(args.interleave),
            bit_order="LSB-first-per-byte",
        )),
        "payload_verification": {
            "checksum": payload_checksum,
            "coded_bytes": 167,
            "coded_bits": 1336,
            "interleave_preserves_length": True,
            "original_metadata": args.metadata,
            "packer_version": "6bit_fields_v1"
        },
        "repeat": int(args.repeat),
        "base_symbol_amp": float(args.base_symbol_amp),
        "psychoacoustic_params": {
            "per_sec_capacity": int(args.per_sec_capacity),
            "amp_safety_factor": 1.0,
            "actual_per_sec_capacity": int(len(filtered_global) / len(chunks)) if len(chunks) > 0 else 0,
            "total_available_slots": int(len(filtered_global)),
            "total_required_slots": int(need),
            "capacity_utilization": float(len(filtered_global) / max(1, need)) if need > 0 else 0.0
        },
        "capacity_policy": {
            "capacity_ok": total_capacity >= required_capacity,
            "final_repeat": int(args.repeat),
            "effective_target_bits": int(num_bits),
            "total_encoded_symbols": int(len(filtered_global)),
            "capacity_warning_issued": total_capacity < required_capacity
        },
        "repetition_distribution": {
            "policy": "round_robin_with_global_shuffle",
            "min_seconds_per_bit": min(len(bit_distribution[bit_idx]["seconds_covered"]) for bit_idx in bit_distribution) if bit_distribution else 0,
            "max_seconds_per_bit": max(len(bit_distribution[bit_idx]["seconds_covered"]) for bit_idx in bit_distribution) if bit_distribution else 0,
            "min_frames_per_bit": min(len(bit_distribution[bit_idx]["frames_covered"]) for bit_idx in bit_distribution) if bit_distribution else 0,
            "max_frames_per_bit": max(len(bit_distribution[bit_idx]["frames_covered"]) for bit_idx in bit_distribution) if bit_distribution else 0,
            "per_bit_summary": {str(bit_idx): bit_distribution[bit_idx] for bit_idx in bit_distribution}
        },
        "clipping_telemetry": {
            "global_stats": {
                "max_peak_across_all_seconds": max(t["pre_clip_peak"] for t in clipping_telemetry) if clipping_telemetry else 0.0,
                "total_clipped_samples": sum(t["clipped_samples"] for t in clipping_telemetry),
                "total_samples": sum(t["total_samples"] for t in clipping_telemetry),
                "worst_clipping_ratio": max(t["clipping_ratio"] for t in clipping_telemetry) if clipping_telemetry else 0.0,
                "worst_headroom_db": min(t["headroom_db"] for t in clipping_telemetry) if clipping_telemetry else 0.0,
                "avg_slot_amp_scale": sum(t["avg_slot_amp_scale"] for t in clipping_telemetry) / max(1, len(clipping_telemetry)) if clipping_telemetry else 0.0
            },
            "per_second": clipping_telemetry
        },
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


