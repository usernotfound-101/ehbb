#!/usr/bin/env bash
# Sets up a Python venv with everything cipher_transformer.py needs.
# Run once (ideally somewhere with internet, e.g. Ada's login node):
#
#   bash setup_env.sh          # GPU node: torch pinned to a CUDA build the
#                               #   cluster's driver can actually run (see below)
#   bash setup_env.sh --cpu    # force CPU-only torch (e.g. testing on a laptop)
#
# IMPORTANT: don't just `pip install torch` unpinned on a cluster. pip
# resolves to the newest build, which can require a newer CUDA runtime than
# the node's NVIDIA driver supports -- and PyTorch does NOT error out when
# that happens, it silently falls back to CPU (torch.cuda.is_available() ==
# False), so training "works" but is 10-100x slower and you won't notice
# until you're watching an 8-hour job crawl. This is exactly what happened on
# Ada's 2080 Ti nodes: driver max is CUDA 12.8, unpinned pip grabbed a build
# needing newer. Pinned to cu121 below (comfortably <= 12.8; CUDA is backward
# compatible, so a build for an OLDER CUDA than the driver supports is fine).
# If your node's driver is even older, check `nvidia-smi`'s "CUDA Version: "
# line and drop to --index-url .../whl/cu118 accordingly.
#
# Then each time you actually run training:
#
#   source .venv/bin/activate
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...        # only needed for --push-to-hub
#   python cipher_transformer.py --config C1_base --epochs 40
set -euo pipefail

VENV_DIR=".venv"
CPU_ONLY=0
[ "${1:-}" = "--cpu" ] && CPU_ONLY=1

# On Ada (or any module-based cluster) you may first need e.g.:
#   module load python/3.10 cuda/12.1
# Uncomment/adjust if `python3` isn't already on PATH or doesn't see a GPU toolchain.
# module load python/3.10

echo "creating venv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

if [ "${CPU_ONLY}" -eq 1 ]; then
    echo "installing CPU-only torch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    CUDA_TAG="cu121"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi driver info:"
        nvidia-smi | grep -i "CUDA Version" || true
    fi
    echo "installing torch pinned to ${CUDA_TAG} (see the comment above -- change this tag if needed)"
    pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

pip install -r requirements.txt

echo
echo "done. activate with: source ${VENV_DIR}/bin/activate"
CUDA_OK=$(python3 -c "import torch; print(int(torch.cuda.is_available()))")
python3 -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if [ "${CPU_ONLY}" -eq 0 ] && [ "${CUDA_OK}" -eq 0 ]; then
    echo "WARNING: requested a GPU build but CUDA is NOT available -- training will silently"
    echo "         run on CPU and be far too slow. Check the CUDA_TAG pin above against"
    echo "         nvidia-smi's reported \"CUDA Version\" and re-run setup_env.sh."
fi
