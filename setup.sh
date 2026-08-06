#!/usr/bin/env bash
#
# Idempotent setup for the Backyard Wildlife trail-cam processor.
# Safe to re-run: pip installs are no-ops when satisfied and the MegaDetector
# weights are only downloaded if not already cached.
set -euo pipefail

cd "$(dirname "$0")"

# 1. Install CPU-only PyTorch wheels first. This VM has no GPU, so pulling the
#    default CUDA build would waste ~1GB and provide no benefit. Installing
#    torch here means the unpinned `torch` requirement of PytorchWildlife is
#    already satisfied and pip will not replace it with a CUDA build.
python3 -m pip install --user --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision torchaudio

# 2. Install the remaining Python dependencies.
python3 -m pip install --user -r requirements.txt

# 3. Pre-download the MegaDetector v6 (MDV6-yolov9-c) weights into the torch hub
#    cache so runs don't fetch ~50MB from Zenodo and work without network access.
#
#    Work around a PytorchWildlife 1.3.0 bug: MegaDetectorV6 checks its cache for
#    MODEL_NAME ("MDV6b-yolov9-c.pt", note the "b"), but wget saves the Zenodo
#    file as "MDV6-yolov9-c.pt". The check never matches, so the library
#    re-downloads on every instantiation (leaving "... (1).pt" copies) and fails
#    when offline. Downloading directly to the exact MODEL_NAME path makes the
#    library find the cached file and skip the download entirely.
CKPT_DIR="$(python3 -c 'import torch, os; print(os.path.join(torch.hub.get_dir(), "checkpoints"))')"
MODEL_PATH="$CKPT_DIR/MDV6b-yolov9-c.pt"
MODEL_URL="https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
mkdir -p "$CKPT_DIR"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading MegaDetector v6 weights to $MODEL_PATH ..."
  curl -fL --retry 4 --retry-delay 4 -o "$MODEL_PATH" "$MODEL_URL"
else
  echo "MegaDetector v6 weights already cached at $MODEL_PATH"
fi

# Verify the weights load from cache (no download, no network required).
python3 - <<'PY'
from PytorchWildlife.models import detection as pw_detection

pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
print("MegaDetector v6 weights ready.")
PY

echo "Backyard Wildlife environment setup complete."
