"""MFCC feature extraction.

A single source of truth for "what does a feature tensor look like".
Both the training pipeline (Python) and the unit tests rely on the
functions defined here.  The Arduino firmware reproduces the same
computation in C++ using CMSIS-DSP — see
``arduino/HeyJarvisWakeWord/feature_provider.cpp``.
"""
from __future__ import annotations

import numpy as np

from config import (
    F_MAX,
    F_MIN,
    FEATURE_SHAPE,
    N_FFT,
    N_MELS,
    N_MFCC,
    NUM_FRAMES,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
    WINDOW_STRIDE_SAMPLES,
)


def compute_mfcc(audio: np.ndarray) -> np.ndarray:
    """Return MFCCs with shape ``FEATURE_SHAPE``.

    The function uses ``librosa`` internally but is wrapped so callers
    do not have to know about hop_length / n_fft semantics.  Output is
    a float32 array shaped ``(NUM_FRAMES, N_MFCC, 1)`` ready to be fed
    to the Keras model.
    """
    # Local import keeps unit tests fast when librosa isn't installed.
    import librosa

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=WINDOW_STRIDE_SAMPLES,
        win_length=WINDOW_SIZE_SAMPLES,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        center=False,
    )
    # librosa returns (n_mfcc, n_frames); transpose to (n_frames, n_mfcc)
    mfcc = mfcc.T
    # Clip / pad time axis to NUM_FRAMES.
    if mfcc.shape[0] > NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES]
    elif mfcc.shape[0] < NUM_FRAMES:
        pad = NUM_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    mfcc = mfcc[..., np.newaxis].astype(np.float32, copy=False)
    assert mfcc.shape == FEATURE_SHAPE, (mfcc.shape, FEATURE_SHAPE)
    return mfcc


def normalize_features(features: np.ndarray, mean: float | None = None,
                       std: float | None = None) -> tuple:
    """Zero-mean / unit-variance normalisation across the whole dataset.

    If ``mean``/``std`` are ``None`` they are computed from ``features``.
    Returns ``(normalised, mean, std)``.
    """
    if mean is None:
        mean = float(features.mean())
    if std is None:
        std = float(features.std() + 1e-8)
    normalised = (features - mean) / std
    return normalised.astype(np.float32, copy=False), mean, std
