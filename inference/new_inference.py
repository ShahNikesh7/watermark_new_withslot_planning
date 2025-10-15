#!/usr/bin/env python3
"""
Aligned inference script that:
- Loads decode checkpoint and uses cfg params (n_fft, hop, target_bits, base_symbol_amp, RS/interleave, planner)
- Resamples to 22050 Hz, splits full audio into 1s chunks, embeds across entire audio
- Computes BER vs embedded bits and tries RS decode of first codeword
- Saves watermarked audio and a side-by-side spectrogram image (original vs watermarked)
"""

from __future__ import annotations
import argparse
import os
import sys
import torch
import torchaudio
import torch.nn.functional as F

# Make project root importable when running this script from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.psychoacoustic import mel_proxy_threshold
from pipeline.ingest_and_chunk import (
    rs_encode_167_125,
    rs_decode_167_125,
    interleave_bytes,
    deinterleave_bytes,
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


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # [C,T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-9)
    wav, sr = _resample_if_needed(wav, sr)
    return wav, sr


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


@torch.no_grad()
def _fast_gpu_plan_slots(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int) -> tuple[list[tuple[int,int]], torch.Tensor]:
    X = model.stft(x_wave)  # [1,2,F,T]
    B, _, F, T = X.shape
    assert B == 1
    mag = torch.sqrt(torch.clamp(X[:,0]**2 + X[:,1]**2, min=1e-6))
    thr = mel_proxy_threshold(X, n_mels=64)
    score = mag / (thr + 1e-6)
    k = int(min(target_bits, F*T))
    _vals, idx = score[0].flatten().topk(k)
    f = (idx // T).to(torch.int64)
    t = (idx % T).to(torch.int64)
    slots: list[tuple[int,int]] = [(int(f[i]), int(t[i])) for i in range(k)]
    amp = thr[0].flatten()[idx]
    med = torch.median(amp)
    amp = amp / (med + 1e-9)
    return slots, amp.detach()


@torch.no_grad()
def _mg_plan_slots(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int, n_fft: int) -> tuple[list[tuple[int,int]], torch.Tensor]:
    X = model.stft(x_wave)
    slots, amp = allocate_slots_and_amplitudes(X, TARGET_SR, n_fft, target_bits, amp_safety=1.0)
    if isinstance(amp, torch.Tensor):
        amp_t = amp
    else:
        import numpy as np  # local import
        if isinstance(amp, np.ndarray):
            amp_t = torch.from_numpy(amp)
        else:
            amp_t = torch.ones(len(slots), dtype=torch.float32)
    return slots, amp_t.to(x_wave.device)


def _encode_payload_bits_from_text(text: str, rs_payload_bytes: int, interleave_depth: int, target_bits: int, device: torch.device) -> torch.Tensor:
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) >= rs_payload_bytes:
        payload_bytes = raw[:rs_payload_bytes]
    else:
        payload_bytes = raw + bytes(rs_payload_bytes - len(raw))
    rs_encoded = rs_encode_167_125(payload_bytes)
    interleaved = interleave_bytes(rs_encoded, interleave_depth)
    bits = []
    for b in interleaved:
        for k in range(8):
            bits.append((b >> k) & 1)
    bit_t = torch.tensor(bits, dtype=torch.long, device=device)
    if bit_t.numel() >= target_bits:
        bit_t = bit_t[:target_bits]
    else:
        pad = torch.zeros(target_bits - bit_t.numel(), dtype=torch.long, device=device)
        bit_t = torch.cat([bit_t, pad], dim=0)
    return bit_t.unsqueeze(0)


def _decode_payload_bits(bits: torch.Tensor, interleave_depth: int, expected_payload_bytes: int) -> bytes:
    if bits.dim() == 2:
        bits = bits[0]
    by = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        b = 0
        for k in range(8):
            b |= (int(bits[i + k]) & 1) << k
        by.append(b)
    deint = deinterleave_bytes(bytes(by), interleave_depth)
    try:
        decoded = rs_decode_167_125(deint)
        return decoded[:expected_payload_bytes]
    except Exception:
        return bytes(expected_payload_bytes)


def _save_spectrogram_side_by_side(orig: torch.Tensor, wm: torch.Tensor, n_fft: int, hop: int, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib not available; skipping spectrogram image")
        return
    # orig, wm: [1, T]
    window = torch.hann_window(n_fft, device=orig.device)
    def spec_mag(x: torch.Tensor):
        X = torch.stft(
            x.squeeze(0), n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True, center=False
        )
        return torch.abs(X).clamp_min(1e-9)  # [F,T]

    M1 = spec_mag(orig)
    M2 = spec_mag(wm)
    # Ensure same time frames for side-by-side comparison
    Tm = min(M1.size(-1), M2.size(-1))
    M1 = M1[:, :Tm]
    M2 = M2[:, :Tm]
    
    # Remove any trailing silence/black regions by finding the last non-silent frame
    # Find the last frame with significant energy (above -60 dB threshold)
    energy1 = torch.log10(M1 + 1e-9).max(dim=0)[0]  # Max energy per frame
    energy2 = torch.log10(M2 + 1e-9).max(dim=0)[0]
    max_energy = torch.maximum(energy1, energy2)
    last_active_frame = torch.where(max_energy > -60.0)[0]
    if len(last_active_frame) > 0:
        last_frame = last_active_frame[-1].item() + 1
        M1 = M1[:, :last_frame]
        M2 = M2[:, :last_frame]
    
    # Convert to dB relative to a common reference to avoid extreme ranges
    ref = torch.maximum(M1.max(), M2.max()).clamp_min(1e-6)
    S1 = (20.0 * torch.log10(M1 / ref)).cpu().numpy()
    S2 = (20.0 * torch.log10(M2 / ref)).cpu().numpy()
    vmin, vmax = -80.0, 0.0

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    im0 = axs[0].imshow(S1, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='magma')
    axs[0].set_title('Original')
    im1 = axs[1].imshow(S2, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='magma')
    axs[1].set_title('Watermarked')
    for ax in axs:
        ax.set_xlabel('Frames')
        ax.set_ylabel('Frequency bins')
    
    # Fix colorbar positioning - only attach to the right subplot
    cbar = fig.colorbar(im1, ax=axs[1], shrink=0.8, pad=0.02)
    cbar.set_label('dB (relative)', rotation=270, labelpad=20)
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--out", type=str, default="watermarked.wav")
    parser.add_argument("--spec_out", type=str, default="spectrogram_compare.png")
    parser.add_argument("--payload", type=str, default="ISRC12345678910,ISWC12345678910,duration4:20,RDate10/10/10")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--planner", type=str, default=None, choices=[None, "gpu", "mg"], help="Override planner (default: use checkpoint cfg)")
    args = parser.parse_args()
    
    print(f"Starting inference with checkpoint: {args.ckpt}")
    print(f"Audio file: {args.audio}")
    print(f"Device: {args.device}")

    print("Loading checkpoint...")
    state = torch.load(args.ckpt, map_location=args.device)
    cfg = state.get("cfg", {})
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop", 512))
    base_symbol_amp = float(cfg.get("base_symbol_amp", 0.03))
    target_bits = int(cfg.get("target_bits", 1336))
    rs_payload_bytes = int(cfg.get("rs_payload_bytes", 125))
    rs_interleave_depth = int(cfg.get("rs_interleave_depth", 4))
    planner = args.planner or cfg.get("planner", "mg")
    print(f"Config loaded: n_fft={n_fft}, hop={hop}, target_bits={target_bits}, base_symbol_amp={base_symbol_amp}")

    print("Building model...")
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(args.device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    print("Model loaded successfully")

    print("Loading audio...")
    wav, sr = _load_audio_mono(args.audio)
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected SR after resample: {sr}")
    chunks = _chunk_audio_1s(wav)  # list of [1,T]
    print(f"Audio loaded: {len(chunks)} chunks")

    # Plan per chunk
    planned = []
    total_capacity = 0
    for ch in chunks:
        x = ch.to(args.device).unsqueeze(0)
        if planner == "gpu":
            slots, amp_scale = _fast_gpu_plan_slots(model, x, target_bits)
        else:
            slots, amp_scale = _mg_plan_slots(model, x, target_bits, n_fft)
        S = min(len(slots), target_bits)
        planned.append((x, slots, amp_scale, S))
        total_capacity += S

    rs_code_bits = 167 * 8
    want_bits = max(rs_code_bits, total_capacity)
    payload_bits = _encode_payload_bits_from_text(args.payload, rs_payload_bytes, rs_interleave_depth, want_bits, device=torch.device(args.device))

    wm_chunks = []
    rec_bits_list = []
    cursor = 0
    for (x, slots, amp_scale, S) in planned:
        if S == 0:
            wm_chunks.append(x)
            continue
        bits_i = payload_bits[:, cursor:cursor+S]
        if bits_i.size(1) < S:
            pad = torch.zeros(1, S - bits_i.size(1), dtype=torch.long, device=x.device)
            bits_i = torch.cat([bits_i, pad], dim=1)
        cursor += S

        X = model.stft(x)
        F_, T_ = X.shape[-2], X.shape[-1]
        M_spec = torch.zeros(1, 2, F_, T_, device=x.device)
        amp_vec = amp_scale[:S].to(x.device) if isinstance(amp_scale, torch.Tensor) else torch.ones(S, device=x.device)
        signs = (bits_i * 2 - 1).float() * base_symbol_amp * amp_vec.unsqueeze(0)
        for s, (f, t) in enumerate(slots[:S]):
            M_spec[0, 0, f, t] = signs[0, s]

        x_wm, _ = model.encode(x, M_spec)
        x_wm = torch.clamp(x_wm, -1.0, 1.0)
        M_rec = model.decode(x_wm)
        rec_vals = torch.stack([M_rec[0, 0, f, t] for (f, t) in slots[:S]], dim=0).unsqueeze(0)
        rec_bits = (rec_vals > 0).long()
        rec_bits_list.append(rec_bits)
        wm_chunks.append(x_wm)

    # Concatenate to full audio
    x_full = torch.cat([ch.squeeze(0) for ch in chunks], dim=-1)  # [1,T]
    x_wm_full = torch.cat([ch.squeeze(0) for ch in wm_chunks], dim=-1)  # [1,T]

    # BER and optional RS decode
    all_rec_bits = torch.cat(rec_bits_list, dim=1) if len(rec_bits_list) > 0 else torch.zeros(1, 0, dtype=torch.long, device=torch.device(args.device))
    compare_len = min(all_rec_bits.size(1), payload_bits.size(1))
    ber = (all_rec_bits[:, :compare_len] != payload_bits[:, :compare_len]).float().mean().item() if compare_len > 0 else 1.0
    recovered_text = ""
    if all_rec_bits.size(1) >= rs_code_bits:
        recovered_payload = _decode_payload_bits(all_rec_bits[0, :rs_code_bits].detach().cpu(), rs_interleave_depth, rs_payload_bytes)
        try:
            recovered_text = recovered_payload.decode("utf-8", errors="ignore")
        except Exception:
            recovered_text = ""

    # Save outputs (ensure tensors are detached for CPU/GPU compatibility)
    torchaudio.save(args.out, x_wm_full.detach().cpu(), sample_rate=TARGET_SR, encoding="PCM_S", bits_per_sample=24)
    _save_spectrogram_side_by_side(x_full.to(args.device).detach(), x_wm_full.to(args.device).detach(), n_fft=n_fft, hop=hop, out_path=args.spec_out)

    print(f"BER={ber:.6f}")
    if recovered_text:
        print(f"Recovered payload (utf-8, truncated): {recovered_text[:128]}")
    else:
        print("Recovered payload unavailable (insufficient bits for full RS decode)")
    print(f"Saved watermarked audio to: {args.out}")
    print(f"Saved spectrogram image to: {args.spec_out}")


if __name__ == "__main__":
    main()

