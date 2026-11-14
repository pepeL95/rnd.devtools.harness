#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-quasipilot-harness}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' was not found in PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Error: conda environment '$ENV_NAME' does not exist. Run ./setup.sh first." >&2
  exit 1
fi

echo "Synchronizing conda environment '$ENV_NAME'"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

python -m pip install --upgrade pip setuptools wheel build
python -m pip install --upgrade \
  -e ".[dev]" \
  "deepagents==0.7.38" \
  "langchain==1.5.0" \
  "langchain-core==1.8.0" \
  "langchain-google-genai==4.5.0" \
  "google-genai==3.3.0" \
  "langgraph==1.2.22" \
  "langgraph-checkpoint==4.3.2" \
  "langgraph-prebuilt==1.1.2" \
  "langgraph-sdk==0.4.14"

python -m pip check
python -m pytest -q

echo "Environment '$ENV_NAME' is synchronized and verified."
