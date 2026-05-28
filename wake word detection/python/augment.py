"""On-the-fly audio augmentation used during training.

Each augmentation is a pure function ``np.ndarray -> np.ndarray`` so they
can be composed deterministically (seeded RNG) inside the
``tf.data`` pipeline via ``tf.numpy_function``.
"""
from __future__ import annotations

import numpy as np

from config import CLIP_SAMPLES, SAMPLE_RATE, TRAIN


def random_time_shift(audio: np.ndarray, rng: np.random.Generator,
                      max_shift_ms: int = TRAIN.max_time_shift_ms) -> np.ndarray:
    """Roll the audio left/right by a random number of samples (zero-fill)."""
    max_shift = SAMPLE_RATE * max_shift_ms // 1_000
    if max_shift <= 0:
        return audio
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return audio
    out = np.zeros_like(audio)
    if shift > 0:
        out[shift:] = audio[:-shift]
    else:
        out[:shift] = audio[-shift:]
    return out


def random_gain(audio: np.ndarray, rng: np.random.Generator,
                low_db: float = -6.0, high_db: float = 6.0) -> np.ndarray:
    gain_db = float(rng.uniform(low_db, high_db))
    return (audio * (10 ** (gain_db / 20.0))).astype(np.float32)


def add_noise(audio: np.ndarray, noise: np.ndarray, rng: np.random.Generator,
              snr_db_low: float = 0.0, snr_db_high: float = 20.0) -> np.ndarray:
    """Mix ``noise`` into ``audio`` at a random SNR."""
    if noise.shape[0] < audio.shape[0]:
        reps = audio.shape[0] // noise.shape[0] + 1
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, noise.shape[0] - audio.shape[0] + 1))
    noise = noise[start:start + audio.shape[0]]

    sig_power = float(np.mean(audio ** 2) + 1e-12)
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    snr_db = float(rng.uniform(snr_db_low, snr_db_high))
    scale = float(np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10.0)))))
    return (audio + noise * scale).astype(np.float32)


def augment_clip(audio: np.ndarray, noise_pool: list, rng: np.random.Generator) -> np.ndarray:
    """Compose the full augmentation pipeline used at training time."""
    out = audio
    if rng.random() < TRAIN.augment_shift_prob:
        out = random_time_shift(out, rng)
    if rng.random() < TRAIN.augment_gain_prob:
        out = random_gain(out, rng)
    if noise_pool and rng.random() < TRAIN.augment_noise_prob:
        noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
        out = add_noise(out, noise, rng)
    # Final safety clip — keep within int16 range when later quantized.
    out = np.clip(out, -1.0, 1.0)
    if out.shape[0] != CLIP_SAMPLES:
        if out.shape[0] > CLIP_SAMPLES:
            out = out[:CLIP_SAMPLES]
        else:
            out = np.pad(out, (0, CLIP_SAMPLES - out.shape[0]))
    return out.astype(np.float32, copy=False)
