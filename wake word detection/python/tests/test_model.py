"""Smoke tests for the DS-CNN model."""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from config import FEATURE_SHAPE, NUM_CLASSES
from model import build_model, compile_model


def test_model_input_output_shape():
    m = build_model()
    assert m.input_shape == (None, *FEATURE_SHAPE)
    assert m.output_shape == (None, NUM_CLASSES)


def test_model_forward_pass_returns_probabilities():
    m = compile_model(build_model())
    x = np.random.RandomState(0).randn(2, *FEATURE_SHAPE).astype(np.float32)
    y = m.predict(x, verbose=0)
    assert y.shape == (2, NUM_CLASSES)
    # Softmax outputs should be a valid distribution.
    np.testing.assert_allclose(y.sum(axis=1), np.ones(2), atol=1e-4)
    assert (y >= 0).all() and (y <= 1).all()


def test_model_param_count_is_microcontroller_friendly():
    m = build_model()
    n = m.count_params()
    # We want to stay well under 100k trainable params for the Nano BLE.
    assert n < 200_000, f"model too large: {n} params"
