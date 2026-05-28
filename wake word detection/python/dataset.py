"""Build tf.data pipelines from the raw WAV tree.

This module is responsible for:
  * scanning ``data/raw`` for positives, negatives and background noise,
  * loading each WAV once and padding / cropping to 1 s,
  * running augmentation on the fly,
  * computing MFCCs,
  * producing train / val / test ``tf.data.Dataset`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from audio_utils import fix_length, iter_wav_files, load_wav
from augment import augment_clip
from config import (
    CLIP_SAMPLES,
    FEATURE_SHAPE,
    NEGATIVE_DIR,
    NOISE_DIR,
    POSITIVE_DIR,
    POSITIVE_LABEL_INDEX,
    TRAIN,
)
from features import compute_mfcc


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    feature_mean: float
    feature_std: float
    class_counts: dict
    num_train: int


def _load_clips(root: Path) -> np.ndarray:
    files = list(iter_wav_files(root))
    if not files:
        return np.empty((0, CLIP_SAMPLES), dtype=np.float32)
    out = np.zeros((len(files), CLIP_SAMPLES), dtype=np.float32)
    for i, f in enumerate(tqdm(files, desc=f"load {root.name}", leave=False)):
        try:
            audio = load_wav(f)
        except Exception:
            continue
        out[i] = fix_length(audio)
    return out


def _split_indices(n: int, val_split: float, test_split: float,
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def build_datasets(seed: int = TRAIN.random_seed) -> DatasetBundle:
    positives = _load_clips(POSITIVE_DIR)
    negatives = _load_clips(NEGATIVE_DIR)
    noises = _load_clips(NOISE_DIR)

    if positives.shape[0] == 0:
        raise RuntimeError(
            f"No positive samples found in {POSITIVE_DIR}. "
            "Run `python record_samples.py` first."
        )
    if negatives.shape[0] == 0:
        raise RuntimeError(
            f"No negative samples found in {NEGATIVE_DIR}. "
            "Run `python prepare_dataset.py --speech-commands` first."
        )

    # Build label arrays.
    audio = np.concatenate([positives, negatives], axis=0)
    labels = np.concatenate([
        np.full(positives.shape[0], POSITIVE_LABEL_INDEX, dtype=np.int32),
        np.full(negatives.shape[0], 1 - POSITIVE_LABEL_INDEX, dtype=np.int32),
    ])

    train_idx, val_idx, test_idx = _split_indices(
        len(audio), TRAIN.val_split, TRAIN.test_split, seed
    )

    noise_pool: List[np.ndarray] = [n for n in noises]
    rng = np.random.default_rng(seed)

    # ---- on-the-fly augmentation + MFCC via numpy_function ----
    def _augment_and_extract(clip: np.ndarray, label: np.int32, training: bool):
        clip = clip.astype(np.float32)
        if training:
            clip = augment_clip(clip, noise_pool, rng)
        feats = compute_mfcc(clip)
        return feats.astype(np.float32), np.int32(label)

    def make_ds(idx: np.ndarray, training: bool, batch_size: int) -> tf.data.Dataset:
        clips_split = audio[idx]
        labels_split = labels[idx]

        def gen():
            order = np.arange(len(idx))
            if training:
                rng.shuffle(order)
            for k in order:
                feats, lbl = _augment_and_extract(clips_split[k], labels_split[k], training)
                yield feats, lbl

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=FEATURE_SHAPE, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        if training:
            ds = ds.shuffle(min(1024, len(idx)), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # Compute feature-level mean/std from a small sample of training clips
    # so that the (optional) normalization step is reproducible.
    sample_feats = []
    sample_n = min(256, len(train_idx))
    for k in train_idx[:sample_n]:
        sample_feats.append(compute_mfcc(audio[k]))
    sample_feats = np.stack(sample_feats, axis=0)
    feature_mean = float(sample_feats.mean())
    feature_std = float(sample_feats.std() + 1e-8)

    train_ds = make_ds(train_idx, training=True, batch_size=TRAIN.batch_size)
    val_ds = make_ds(val_idx, training=False, batch_size=TRAIN.batch_size)
    test_ds = make_ds(test_idx, training=False, batch_size=TRAIN.batch_size)

    class_counts = {
        "hey_jarvis": int((labels == POSITIVE_LABEL_INDEX).sum()),
        "background": int((labels == 1 - POSITIVE_LABEL_INDEX).sum()),
    }

    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_counts=class_counts,
        num_train=len(train_idx),
    )
