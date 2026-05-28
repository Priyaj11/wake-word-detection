"""Dataset preparation.

Builds the ``data/raw`` tree used by ``train.py``:

    data/raw/
        hey_jarvis/        # positive samples (user-recorded WAVs)
        negative/          # negative speech samples (random words)
        background_noise/  # ~1 minute clips of pure noise / silence

If the Google Speech Commands dataset is available it is automatically
unpacked into ``negative/`` and ``background_noise/`` to give the
network a strong "everything-that-isn't-Hey-Jarvis" prior.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from audio_utils import iter_wav_files, load_wav, save_wav
from config import (
    DATA_DIR,
    NEGATIVE_DIR,
    NOISE_DIR,
    POSITIVE_DIR,
    SAMPLE_RATE,
    ensure_dirs,
)

SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)

# Speech-commands words to repurpose as the "negative" class.
NEGATIVE_WORDS = (
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog",
    "happy", "house", "marvin", "sheila", "tree", "wow",
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] already downloaded: {dest}")
        return dest
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length", "0"))
        bar = tqdm(total=total, unit="B", unit_scale=True)
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
        bar.close()
    return dest


def _extract(tar_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] already extracted: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)


def import_speech_commands(per_word_limit: int = 250) -> None:
    """Download Speech Commands and copy a subset into negative/noise dirs."""
    ensure_dirs()
    archive = DATA_DIR / "speech_commands.tar.gz"
    extracted = DATA_DIR / "speech_commands"
    _download(SPEECH_COMMANDS_URL, archive)
    _extract(archive, extracted)

    copied_neg = 0
    for word in NEGATIVE_WORDS:
        src = extracted / word
        if not src.exists():
            continue
        files = sorted(src.glob("*.wav"))[:per_word_limit]
        for f in tqdm(files, desc=f"negative/{word}", leave=False):
            dst = NEGATIVE_DIR / f"{word}_{f.stem}.wav"
            if not dst.exists():
                shutil.copyfile(f, dst)
                copied_neg += 1
    print(f"[ok] imported {copied_neg} negative clips")

    bg_src = extracted / "_background_noise_"
    if bg_src.exists():
        copied = 0
        for f in tqdm(sorted(bg_src.glob("*.wav")), desc="background_noise"):
            dst = NOISE_DIR / f.name
            if not dst.exists():
                shutil.copyfile(f, dst)
                copied += 1
        print(f"[ok] imported {copied} noise clips")


def stats() -> None:
    counts = {
        "hey_jarvis": sum(1 for _ in iter_wav_files(POSITIVE_DIR)),
        "negative":   sum(1 for _ in iter_wav_files(NEGATIVE_DIR)),
        "noise":      sum(1 for _ in iter_wav_files(NOISE_DIR)),
    }
    for k, v in counts.items():
        print(f"  {k:>16}: {v} files")
    if counts["hey_jarvis"] < 30:
        print("\n[!] You have very few positive samples.")
        print("    Record at least 50–100 clips with `python record_samples.py`.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--speech-commands",
        action="store_true",
        help="Download Google Speech Commands and use it for the negative class.",
    )
    p.add_argument(
        "--per-word-limit", type=int, default=250,
        help="Number of clips to import per negative word (default: 250).",
    )
    args = p.parse_args()
    ensure_dirs()
    if args.speech_commands:
        import_speech_commands(per_word_limit=args.per_word_limit)
    stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
