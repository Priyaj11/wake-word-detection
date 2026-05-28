# Dataset guide

Wake-word detection only works as well as the data it was trained on.
This guide explains how to assemble a small but effective dataset for the
"Hey Jarvis" detector.

## Folder layout

```
data/raw/
├── hey_jarvis/         # Positive — every clip you record yourself
├── negative/           # Negative speech — any words that AREN'T the wake word
└── background_noise/   # Pure noise / silence (room tone, music, traffic)
```

`python/prepare_dataset.py --speech-commands` will fill `negative/` and
`background_noise/` automatically using Google Speech Commands v2.

## Positive samples

Goal: ≥ 50 clips, ideally 100+.

Recording tips:

* 1 second per clip — say "Hey Jarvis" once, naturally, within the window.
* Vary distance from the mic (10–60 cm).
* Vary volume (whisper → conversational → loud).
* Record at different times of day to capture different room acoustics.
* If multiple people will use it, record at least 20 clips per voice.

Run `python/record_samples.py` to capture clips directly into
`data/raw/hey_jarvis/` with consistent normalisation.

## Negative samples

You want roughly 5–10× more negative clips than positive — Speech Commands
gives ~7 500 clips across 30 words for free.  If you also have a clean source
of conversational speech (e.g. LibriSpeech), feel free to copy 1-second slices
into `data/raw/negative/` — the more variety the better.

## Background noise

Record ~1 minute clips of:

* Quiet room tone.
* Typing / mouse clicks.
* Music at low volume.
* HVAC noise.
* TV in another room.

These are mixed in at random SNRs by `augment.py`, which makes the
on-device detector dramatically more robust without enlarging the model.

## File format

All WAVs should be:

* 16 kHz mono PCM (the loader will resample if needed)
* float32 in the range [-1, 1] (the loader handles conversion)
* Any duration — the loader pads / center-crops to exactly 1 second

## Verification

After preparing data:

```bash
python prepare_dataset.py            # prints counts per class
```

You should see something like:

```
        hey_jarvis: 78 files
          negative: 7500 files
             noise: 6 files
```
