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
  `output/low_confidence`, or `output/empties` based on the max animal confidence,
  renamed with a `conf<NN>_<camera>_<timestamp>` prefix,
- saves an annotated copy (bounding boxes + labels) to `output/annotated/`,
- appends a row to `backyard_wildlife_log.csv`.

### Tuning detections

`DETECTION_CONF_THRESHOLD` (top of `process_wildlife.py`, default `0.1`) controls
the lowest confidence a detection needs to be recorded. Lower it to capture more
faint/distant animals (fewer false empties); raise it toward `0.2` to cut false
positives.
