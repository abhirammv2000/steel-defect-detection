# Steel Defect Detection

A trained, served segmentation model for the [Severstal Steel Defect
Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
dataset. The task is pixel-level segmentation across 4 defect classes with
RLE-encoded masks, served with three separate outputs: segmentation, anomaly
detection, and an objective grading score.

## What's real here, and what isn't

This is scoped honestly. It is not a full reproduction of a Kaggle
leaderboard result.

- No GPU on this machine (checked with `nvidia-smi` before deciding scope).
- Training ran on 300 of the 6,666 annotated images (a random subset), at
  half native resolution (128x800, down from 256x1600), for 5 epochs, on
  CPU. The run is recorded in `models/training_run.json`: final validation
  Dice was 0.235, peak 0.301 at epoch 3 (see the note below about the
  recorded training time).
- The model is genuinely trained and working, with modest accuracy. That's
  the expected result of a small, fast, CPU-only run. A full run (all
  ~12.5k images, full resolution, more epochs) would need real GPU compute,
  which this environment doesn't have. Scaling up is a config change
  (`--subset-size`, `--epochs` in `training/train.py`), not a rewrite.
- All three capabilities are real and independently testable:
  - Segmentation: `POST /predict` returns a per-class predicted mask, as
    area coverage and confidence rather than the full-resolution mask, to
    keep responses small.
  - Anomaly detection: `defect_present`, derived from whether any class's
    predicted mask is non-empty.
  - Objective grading: a deterministic severity score (`app/grading.py`).
    The formula and three worked examples are in that file's docstring and
    locked down by `tests/test_grading.py`.

### A bug caught while writing this up

`models/training_run.json`'s epoch 5 shows `epoch_seconds: 42023.8` (about
11.7 hours), against 206-334 seconds for epochs 1-4. That's not real
training time. It's `time.time()` capturing a real wall-clock gap in the
session this ran in (the process kept running, but there was a long pause
between epoch 4 finishing and epoch 5's result being read back). The actual
compute for epoch 5 matched epochs 3-4. Real total training time: about
1,022s for epochs 1-4 plus roughly 200s for epoch 5, so 20-21 minutes total,
not the `total_training_seconds` figure the JSON literally contains. That
figure is left uncorrected in the file (it's real output from the run) with
this explanation instead.

## Architecture

```
severstal-steel-defect-detection/train.csv, train_images/  (real Kaggle data)
            |
            v
   app/rle.py            RLE encode/decode (Severstal's column-major format)
            |
            v
   app/dataset.py         PyTorch Dataset, builds a (4, H, W) multi-class
                          mask per image, resized for CPU training time
            |
            v
   training/train.py      U-Net (ResNet18 encoder, ImageNet-pretrained) +
                          Dice+BCE loss, Dice tracked per epoch
            |
            v
   models/steel_defect_unet.pt   trained weights (gitignored, see below)
            |
            v
   app/main.py + app/grading.py    FastAPI service: POST /predict returns
                          segmentation + anomaly detection + objective grading
```

The model weights (`models/steel_defect_unet.pt`, ~57MB) are gitignored. A
trained binary doesn't belong in git history, and `training/train.py`
reproduces it deterministically (`--seed`) from the same data.

## Running it

```bash
python -m venv venv
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip install -r requirements.txt

# train (real data must already be under severstal-steel-defect-detection/)
venv\Scripts\python -m training.train --subset-size 300 --epochs 5

# serve
venv\Scripts\uvicorn app.main:app --port 8020
curl -X POST http://127.0.0.1:8020/predict -F "file=@severstal-steel-defect-detection/test_images/0000f269f.jpg"
```

Real output from that exact command against that exact image:

```json
{
  "defect_present": true,
  "classes": [
    {"class_id": 1, "defect_present": false, "area_fraction": 0.0, "max_confidence": 0.4027},
    {"class_id": 2, "defect_present": false, "area_fraction": 0.0, "max_confidence": 0.2456},
    {"class_id": 3, "defect_present": true, "area_fraction": 0.14527, "max_confidence": 0.9985},
    {"class_id": 4, "defect_present": false, "area_fraction": 0.0, "max_confidence": 0.3888}
  ],
  "grade": {"score": 100.0, "letter": "D", "total_area_fraction": 0.14527, "num_defect_classes_present": 1}
}
```

## Tests

```bash
venv\Scripts\python -m pytest tests/ -v
```

12 tests: RLE encode/decode checked against both synthetic cases and a real
round trip against an actual competition CSV row (this is the only way to
catch a wrong convention assumption, like row-major vs. the real
column-major format); the grading formula's three worked examples, checked
by direct computation (an earlier draft of the threshold logic had a real
off-by-one at the grade boundaries, caught this way before it shipped); and
an inference smoke test against a real held-out test image using the actual
trained model (skipped if the model artifact isn't present, for example on
a fresh clone before training has run).

## Known limitations

- Trained only on images that have at least one labeled defect (6,666 of
  the dataset's 12,568 train images). The model has never seen a fully
  defect-free sheet at the whole-image level, only defect-free background
  pixels within positive images. False-positive rate on genuinely clean
  sheets is untested.
- The grading formula weights all four defect classes equally per unit
  area. A real manufacturing deployment would likely weight by
  cost-to-repair or safety impact, which needs domain expertise this
  project doesn't have. That's stated plainly in `app/grading.py`'s
  docstring.
- The modest Dice score (0.235-0.30) is the expected result of a small
  CPU-scoped training run, not a ceiling on what this approach could do
  with real GPU compute and the full dataset.
