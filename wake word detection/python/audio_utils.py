"""Audio I/O and lightweight DSP helpers.

These utilities are deliberately framework-free (only numpy + soundfile +
scipy) so they can be used from the dataset preparation scripts, the
augmentation pipeline and the unit tests without pulling TensorFlow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import CLIP_SAMPLES, SAMPLE_RATE


def load_wav(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as a mono float32 array resampled to ``target_sr``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file has zero samples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.size == 0:
        raise ValueError(f"empty audio file: {path}")
    if data.ndim > 1:                    # stereo → mono
        data = data.mean(axis=1)
    if sr != target_sr:
        # Use polyphase resampling — high quality, fast, no SciPy warnings.
        from math import gcd
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data.astype(np.float32, copy=False)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save a float32/-1..1 array as a 16-bit PCM mono WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def fix_length(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad with zeros or center-crop ``audio`` to exactly ``length`` samples."""
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        start = (audio.shape[0] - length) // 2
        return audio[start:start + length]
    pad = length - audio.shape[0]
    left = pad // 2
    right = pad - left
    return np.pad(audio, (left, right), mode="constant")


def rms_normalize(audio: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    """Scale ``audio`` so its RMS matches ``target_dbfs`` (in dB relative to 1.0)."""
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms < 1e-6:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    out = audio * gain
    # Prevent hard clipping after normalization.
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32, copy=False)


def iter_wav_files(root: str | Path) -> Iterable[Path]:
    """Yield every .wav file under ``root`` (recursive, sorted)."""
    root = Path(root)
    if not root.exists():
        return
    yield from sorted(root.rglob("*.wav"))
