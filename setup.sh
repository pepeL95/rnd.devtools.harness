#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="quasipilot-harness"
PYTHON_VERSION="3.13"

echo "Setting up conda environment '$ENV_NAME' with Python $PYTHON_VERSION"

# Ensure conda/mamba is available and initialize shell integration
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif command -v mamba >/dev/null 2>&1; then
  if mamba info --base >/dev/null 2>&1; then
    CONDA_BASE=$(mamba info --base | awk 'NR==1{print $1}') || true
    if [ -n "$CONDA_BASE" ]; then
      # shellcheck source=/dev/null
      source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
  fi
else
  echo "Error: neither 'conda' nor 'mamba' found in PATH. Install Miniconda/Anaconda or Mamba first." >&2
  exit 1
fi

# Create environment if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists — skipping creation."
else
  echo "Creating conda environment '$ENV_NAME'..."
  if command -v mamba >/dev/null 2>&1; then
    mamba create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
  else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
  fi
fi

echo "Activating environment '$ENV_NAME'"
conda activate "$ENV_NAME"

echo "Upgrading packaging tools (pip, setuptools, wheel)"
python -m pip install --upgrade pip setuptools wheel build

# Install the project and CLI entrypoint. Prefer editable install to ease development.
if [ -f "pyproject.toml" ]; then
  echo "Installing project from local source (including optional 'dev' extras)"
  python -m pip install -e ".[dev]"
else
  echo "pyproject.toml not found; installing core runtime and CLI packages manually"
  python -m pip install \
    "chromadb>=1.0" \
    "deepagents>=0.6.12" \
    "langchain>=1.3.11,<2" \
    "langchain-google-genai>=4.2.5" \
    "langgraph>=1.2.5" \
    "pydantic>=2.0" \
    "python-dotenv>=1.0" \
    "textual>=6.0" \
    "tiktoken>=0.12"
fi

echo
echo "Done. To start using the harness and CLI:"
echo "  1) Activate the environment: source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "  2) Ensure .env contains GOOGLE_API_KEY or GEMINI_API_KEY"
echo "  3) Run the CLI using: quasipilot"
echo
echo "If you want the environment to be created fresh, remove it first with: conda env remove -n $ENV_NAME"
