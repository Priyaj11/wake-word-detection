"""Evaluate a trained model on the held-out test split.

Produces:
  artifacts/logs/confusion_matrix.png
  artifacts/logs/metrics.json   (accuracy, precision, recall, F1, ROC-AUC)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    CLASS_NAMES,
    KERAS_PATH,
    LOGS_DIR,
    POSITIVE_LABEL_INDEX,
    TRAIN,
    ensure_dirs,
)
from dataset import build_datasets


def _evaluate(model: tf.keras.Model, ds: tf.data.Dataset) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_true, y_prob = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_prob.append(probs)
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)
    pos_prob = y_prob[:, POSITIVE_LABEL_INDEX]

    pos_true = (y_true == POSITIVE_LABEL_INDEX).astype(np.int32)
    pos_pred = (y_pred == POSITIVE_LABEL_INDEX).astype(np.int32)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(pos_true, pos_pred, zero_division=0)),
        "recall": float(recall_score(pos_true, pos_pred, zero_division=0)),
        "f1": float(f1_score(pos_true, pos_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(pos_true, pos_prob))
            if len(np.unique(pos_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred,
                                        target_names=CLASS_NAMES, zero_division=0),
    }


def _plot_confusion_matrix(cm: list, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=str(KERAS_PATH))
    parser.add_argument("--seed", type=int, default=TRAIN.random_seed)
    args = parser.parse_args()

    ensure_dirs()
    model = tf.keras.models.load_model(args.model)
    bundle = build_datasets(seed=args.seed)
    metrics = _evaluate(model, bundle.test)
    print(metrics["report"])
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"F1      : {metrics['f1']:.4f}")

    out_json = LOGS_DIR / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    _plot_confusion_matrix(metrics["confusion_matrix"],
                           LOGS_DIR / "confusion_matrix.png")
    print(f"[ok] metrics written to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
