"""Centralized configuration for the Hey-Jarvis wake-word pipeline.

Every constant the Python pipeline depends on lives here so that the
training, evaluation, conversion and on-device feature extraction stay
in lock-step.  The Arduino firmware mirrors the same values in
``arduino/HeyJarvisWakeWord/config.h``.  If you change a value here,
update the C++ header as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
POSITIVE_DIR: Path = RAW_DATA_DIR / "hey_jarvis"
NEGATIVE_DIR: Path = RAW_DATA_DIR / "negative"
NOISE_DIR: Path = RAW_DATA_DIR / "background_noise"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
TFLITE_PATH: Path = MODELS_DIR / "hey_jarvis_int8.tflite"
KERAS_PATH: Path = MODELS_DIR / "hey_jarvis.keras"
C_ARRAY_PATH: Path = PROJECT_ROOT / "arduino" / "HeyJarvisWakeWord" / "model.cpp"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000           # 16 kHz mono — matches PDM mic
CLIP_DURATION_MS: int = 1_000       # 1 second window
CLIP_SAMPLES: int = SAMPLE_RATE * CLIP_DURATION_MS // 1_000

# ---------------------------------------------------------------------------
# Feature extraction (MFCC)
# ---------------------------------------------------------------------------
WINDOW_SIZE_MS: int = 30
WINDOW_STRIDE_MS: int = 20
WINDOW_SIZE_SAMPLES: int = SAMPLE_RATE * WINDOW_SIZE_MS // 1_000   # 480
WINDOW_STRIDE_SAMPLES: int = SAMPLE_RATE * WINDOW_STRIDE_MS // 1_000  # 320
N_MFCC: int = 40
N_FFT: int = 512
N_MELS: int = 40
F_MIN: float = 20.0
F_MAX: float = 4_000.0   # speech band

NUM_FRAMES: int = 1 + (CLIP_SAMPLES - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES  # 49
FEATURE_SHAPE: tuple = (NUM_FRAMES, N_MFCC, 1)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES: tuple = ("background", "hey_jarvis")
NUM_CLASSES: int = len(CLASS_NAMES)
POSITIVE_LABEL_INDEX: int = CLASS_NAMES.index("hey_jarvis")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.15
    test_split: float = 0.10
    random_seed: int = 42
    early_stopping_patience: int = 6
    lr_plateau_patience: int = 3
    positive_class_weight: float = 1.5
    augment_noise_prob: float = 0.6
    augment_shift_prob: float = 0.5
    augment_gain_prob: float = 0.5
    max_time_shift_ms: int = 100


TRAIN = TrainConfig()

# ---------------------------------------------------------------------------
# Inference / sliding window (mirrored on device)
# ---------------------------------------------------------------------------
DETECTION_THRESHOLD: float = 0.90
SUPPRESSION_MS: int = 1_500
AVERAGING_WINDOW_MS: int = 1_000
MIN_DETECTIONS_IN_WINDOW: int = 2


def ensure_dirs() -> None:
    """Create all working directories used by the pipeline."""
    for d in (
        DATA_DIR, RAW_DATA_DIR, POSITIVE_DIR, NEGATIVE_DIR, NOISE_DIR,
        ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
