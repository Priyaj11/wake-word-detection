"""End-to-end training script.

Usage
-----
$ python train.py                   # train with defaults
$ python train.py --epochs 30       # override

Artifacts produced under ``artifacts/``:
  models/hey_jarvis.keras    — full-precision Keras model
  logs/training_history.json — per-epoch metrics
  logs/training.log          — text log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    KERAS_PATH,
    LOGS_DIR,
    MODELS_DIR,
    POSITIVE_LABEL_INDEX,
    TRAIN,
    ensure_dirs,
)
from dataset import build_datasets
from model import build_model, compile_model


def _setup_logging() -> logging.Logger:
    ensure_dirs()
    log_path = LOGS_DIR / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("train")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.learning_rate)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-ds-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=TRAIN.random_seed)
    args = parser.parse_args()

    log = _setup_logging()
    _set_seed(args.seed)
    ensure_dirs()

    log.info("Loading datasets...")
    bundle = build_datasets(seed=args.seed)
    log.info("Class counts: %s", bundle.class_counts)
    log.info("Train clips: %d", bundle.num_train)
    log.info("Feature mean: %.4f, std: %.4f", bundle.feature_mean, bundle.feature_std)

    log.info("Building DS-CNN model (width=%d, ds_blocks=%d)...",
             args.width, args.num_ds_blocks)
    model = build_model(width=args.width, num_ds_blocks=args.num_ds_blocks)
    compile_model(model, learning_rate=args.lr)
    model.summary(print_fn=log.info)

    total = sum(bundle.class_counts.values())
    pos = bundle.class_counts["hey_jarvis"]
    neg = bundle.class_counts["background"]
    # Inverse-frequency class weights, then bias the positive class slightly.
    base_pos = total / (2.0 * max(pos, 1))
    base_neg = total / (2.0 * max(neg, 1))
    class_weight = {
        POSITIVE_LABEL_INDEX: base_pos * TRAIN.positive_class_weight,
        1 - POSITIVE_LABEL_INDEX: base_neg,
    }
    log.info("Class weights: %s", class_weight)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=TRAIN.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=TRAIN.lr_plateau_patience,
            min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(KERAS_PATH),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    log.info("Starting training for up to %d epochs...", args.epochs)
    history = model.fit(
        bundle.train,
        epochs=args.epochs,
        validation_data=bundle.val,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    log.info("Evaluating on held-out test set...")
    metrics = model.evaluate(bundle.test, return_dict=True, verbose=0)
    log.info("Test metrics: %s", metrics)

    out = {
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "class_counts": bundle.class_counts,
        "feature_mean": bundle.feature_mean,
        "feature_std": bundle.feature_std,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
    }
    (LOGS_DIR / "training_history.json").write_text(json.dumps(out, indent=2))
    log.info("Best model saved to %s", KERAS_PATH)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
