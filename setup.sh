#!/usr/bin/env bash
#
# Idempotent setup for the Backyard Wildlife trail-cam processor.
# Safe to re-run: pip installs are no-ops when satisfied and the MegaDetector
# weights are only downloaded if not already cached.
set -euo pipefail

cd "$(dirname "$0")"

# Decide where pip installs. Inside an active virtualenv (the normal local
# workflow) we install into the venv. On a bare system Python (e.g. the Cloud
# Agent VM, which is PEP 668 "externally managed") we fall back to --user so we
# don't fight the system packages. Using --user *inside* a venv is an error, so
# this detection keeps setup.sh working both locally and in the cloud.
if python3 -c 'import sys, os; raise SystemExit(0 if (sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ) else 1)'; then
  echo "Virtualenv detected -> installing into it."
  PIP_TARGET_FLAG=""
else
  echo "No virtualenv detected -> installing into the user site (--user)."
  PIP_TARGET_FLAG="--user"
fi

# 1. Ensure PyTorch is present. If torch is already installed (e.g. a GPU build
#    you set up locally), leave it alone. Otherwise install CPU-only wheels by
#    default so a machine without a GPU (like the Cloud Agent VM) doesn't pull
#    ~1GB of unusable CUDA libraries. Override the index for a GPU build, e.g.
#    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup.sh
if python3 -c 'import torch' 2>/dev/null; then
  echo "torch already installed ($(python3 -c 'import torch; print(torch.__version__)')) -> skipping."
else
  python3 -m pip install $PIP_TARGET_FLAG \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    torch torchvision torchaudio
fi

# 2. Install the remaining Python dependencies.
python3 -m pip install $PIP_TARGET_FLAG -r requirements.txt

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
