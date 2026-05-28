"""Tests for the MFCC feature extractor."""
from __future__ import annotations

import numpy as np
import pytest

from config import CLIP_SAMPLES, FEATURE_SHAPE, NUM_FRAMES, N_MFCC, SAMPLE_RATE
from features import compute_mfcc, normalize_features


def _tone(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_shape_and_dtype():
    audio = _tone()
    feats = compute_mfcc(audio)
    assert feats.shape == FEATURE_SHAPE
    assert feats.dtype == np.float32


def test_feature_shape_constants_are_consistent():
    # Ensures the config-derived shape matches the doc'd 49 × 40 layout.
    assert FEATURE_SHAPE == (NUM_FRAMES, N_MFCC, 1)
    assert NUM_FRAMES == 49
    assert N_MFCC == 40


def test_feature_is_deterministic_for_same_input():
    audio = _tone()
    a = compute_mfcc(audio)
    b = compute_mfcc(audio)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_short_audio_is_padded():
    # Pass only 0.5 s of audio — extractor should still emit FEATURE_SHAPE.
    audio = _tone(duration_s=0.5)
    # Pad to 1 s like the dataset loader would.
    if audio.shape[0] < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - audio.shape[0]))
    feats = compute_mfcc(audio)
    assert feats.shape == FEATURE_SHAPE


def test_normalize_features_returns_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((4, *FEATURE_SHAPE)).astype(np.float32) * 5 + 3
    norm, mean, std = normalize_features(feats)
    assert pytest.approx(float(norm.mean()), abs=1e-5) == 0.0
    assert pytest.approx(float(norm.std()), abs=1e-3) == 1.0
    assert isinstance(mean, float) and isinstance(std, float)
