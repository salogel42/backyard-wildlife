# backyard-wildlife

Uses [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps) (MegaDetector v6)
to process images and videos from backyard trail cams: it detects animals, sorts
each file into confidence tiers, writes an annotated copy with bounding boxes,
and logs every detection to a CSV.

## Setup

Local use (recommended — this tool runs against your own image/video library,
so you normally run it on your own machine):

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
./setup.sh
```

`setup.sh` installs PyTorch, the `PytorchWildlife` stack (see `requirements.txt`),
and pre-downloads the MegaDetector v6 (`MDV6-yolov9-c`) weights into the torch
hub cache. It is idempotent and safe to re-run, and it adapts to where it runs:

- If a virtualenv is active, it installs into that venv; otherwise it installs
  into your Python user site (used on the Cursor Cloud Agent VM).
- If `torch` is already installed it is left untouched. Otherwise it installs
  **CPU-only** wheels by default. For an NVIDIA GPU (much faster on large
  batches), install a CUDA build first or point setup at a CUDA index:

  ```bash
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup.sh
  ```

In a Cursor Cloud Agent, `setup.sh` runs automatically via
`.cursor/environment.json` (handy for working on the code, though you'll usually
run detection locally to avoid uploading thousands of files).

## Usage

Drop trail-cam files into per-camera folders under the archive directory, then
run the processor:

```
~/BackyardWildlifeArchive/
  incoming/
    cam1/   <- drop .jpg/.jpeg/.png/.mp4/.avi/.mov here
    cam2/
```

```bash
python3 process_wildlife.py
```

For each file the processor:

- runs MegaDetector v6 (on video it samples ~1 frame/second),
- moves it out of `incoming/` into `output/high_confidence`, `output/med_confidence`,
  `output/low_confidence`, `output/empties`, or `output/false_positives` based on
  the max animal confidence, renamed with a `conf<NN>_<camera>_<timestamp>` prefix,
- saves an annotated copy (bounding boxes + labels) to `output/annotated/`,
- appends a row to `backyard_wildlife_log.csv`.

### Tuning detections

All knobs live at the top of `process_wildlife.py` and can be overridden with
environment variables (no code edit needed):

| Setting | Env var | Default | Purpose |
| --- | --- | --- | --- |
| `DETECTION_CONF_THRESHOLD` | `WILDLIFE_CONF_THRESHOLD` | `0.1` | Lowest confidence recorded. Lower = catch more faint animals (fewer false empties); raise toward `0.2` to cut low-confidence noise. |
| `MODEL_VERSION` | `WILDLIFE_MODEL_VERSION` | `MDV6-yolov9-c` | Detector variant (e.g. `MDV6-yolov9-e` is larger/slower). Note: a bigger model does **not** reliably remove static-background false positives. |
| `RDE_ENABLED` | `WILDLIFE_RDE` | `1` | Repeat-detection elimination on/off (`0` disables). |
| `RDE_MIN_REPEATS` | `WILDLIFE_RDE_MIN_REPEATS` | `10` | A box location must recur in this many photos from a camera to count as background. |
| `RDE_IOU_THRESHOLD` | `WILDLIFE_RDE_IOU` | `0.6` | How overlapping two boxes must be to be "the same location". |

### False positives on fixed scenery (repeat-detection elimination)

A fixed trail cam will repeatedly flag the same static object — a branch, trunk,
or rock — as an "animal" in nearly the same spot across many photos. **Repeat-
detection elimination (RDE)** handles this: within a single run it clusters the
animal boxes from each camera's photos and, for any location that recurs in at
least `RDE_MIN_REPEATS` photos, treats it as background and removes those boxes.
Affected photos are archived to `output/false_positives/` (and logged with
status `Repeat Detection (filtered)`) instead of being mislabeled as animals.
A real animal that merely passes through that spot once is kept, because it does
not recur in the same box across many frames.

RDE needs a batch of photos from the same camera to learn what's static, so run
it over a chunk of images at once rather than one at a time. If it ever hides a
real animal, disable it (`WILDLIFE_RDE=0`) or raise `RDE_MIN_REPEATS`.

To sanity-check tuning on your own images without moving anything, copy a
representative batch into a scratch camera folder and compare runs, e.g.:

```bash
WILDLIFE_RDE=0 python3 process_wildlife.py   # see raw detections first
python3 process_wildlife.py                  # then with RDE, compare false_positives/
```
