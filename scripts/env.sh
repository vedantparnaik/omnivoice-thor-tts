# Source this to activate the OmniVoice-Thor environment.
# The Jetson/CUDA-13 torch wheel dynamically links against NVIDIA math libs
# (NVPL + cu13) that ship as pip packages inside the venv. The dynamic linker
# must see them via LD_LIBRARY_PATH before python starts.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
SP="$VENV/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$SP/nvpl/lib:$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"

# Project on path, plus system TensorRT 10.13 python bindings (ABI-compatible
# with the venv's Python 3.12). The venv is isolated, so we add them explicitly.
export PYTHONPATH="$PROJECT_ROOT:/usr/lib/python3.12/dist-packages:${PYTHONPATH:-}"

# Keep model/cache downloads inside the project for a self-contained deliverable.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/artifacts/hf_cache}"
export OMNIVOICE_ARTIFACTS="$PROJECT_ROOT/artifacts"
