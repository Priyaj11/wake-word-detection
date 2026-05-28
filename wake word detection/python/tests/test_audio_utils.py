"""Tests for the framework-free audio utilities."""
from __future__ import annotations

import numpy as np

from audio_utils import fix_length, rms_normalize
from config import CLIP_SAMPLES, SAMPLE_RATE


def test_fix_length_pads_short_input():
    audio = np.ones(SAMPLE_RATE // 2, dtype=np.float32)
    out = fix_length(audio)
    assert out.shape == (CLIP_SAMPLES,)
    # Padding is zero-filled.
    assert out[0] == 0 or out[-1] == 0


def test_fix_length_crops_long_input():
    audio = np.ones(SAMPLE_RATE * 2, dtype=np.float32)
    out = fix_length(audio)
    assert out.shape == (CLIP_SAMPLES,)


def test_fix_length_passes_through_correct_length():
    audio = np.full(CLIP_SAMPLES, 0.5, dtype=np.float32)
    out = fix_length(audio)
    np.testing.assert_array_equal(out, audio)


def test_rms_normalize_targets_dbfs():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.01
    out = rms_normalize(audio, target_dbfs=-20.0)
    rms = float(np.sqrt(np.mean(out ** 2)))
    expected = 10 ** (-20.0 / 20.0)
    # Allow some slack — peak protection may scale the gain down.
    assert rms <= expected + 1e-3
    assert rms > expected * 0.5
