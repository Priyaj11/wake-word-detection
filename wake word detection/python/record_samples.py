"""Interactive recorder for "Hey Jarvis" positive samples.

Usage
-----
$ python record_samples.py --count 60 --speaker priya

Each recording is normalised and saved under ``data/raw/hey_jarvis``.
Press Enter to start each clip; the script records 1 second of audio.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from audio_utils import rms_normalize, save_wav
from config import CLIP_SAMPLES, POSITIVE_DIR, SAMPLE_RATE, ensure_dirs


def _record_one(duration_s: float = 1.0) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:  # pragma: no cover - optional dep
        print("[error] sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)
    n = int(duration_s * SAMPLE_RATE)
    audio = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def _countdown(seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  GO!    ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60,
                        help="Number of samples to record (default: 60).")
    parser.add_argument("--speaker", type=str, default="user",
                        help="Speaker tag included in file names.")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Clip duration in seconds (default: 1.0).")
    parser.add_argument("--countdown", type=int, default=2,
                        help="Countdown seconds before each clip.")
    args = parser.parse_args()

    ensure_dirs()
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.count} clips of 'Hey Jarvis' (speaker={args.speaker})")
    print("Tip: use varied tones, distances and accents for robustness.\n")

    for i in range(1, args.count + 1):
        input(f"[{i}/{args.count}] Press Enter and say 'Hey Jarvis'...")
        _countdown(args.countdown)
        audio = _record_one(args.duration)
        audio = rms_normalize(audio)
        # Ensure exact length.
        if audio.shape[0] >= CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        else:
            audio = np.pad(audio, (0, CLIP_SAMPLES - audio.shape[0]))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = POSITIVE_DIR / f"{args.speaker}_{ts}_{i:03d}.wav"
        save_wav(out, audio)
        print(f"  saved {out.name}\n")

    print(f"[ok] recorded {args.count} clips into {POSITIVE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
