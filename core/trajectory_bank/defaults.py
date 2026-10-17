"""Developer-overridable embedding default, shared by the CLI, driver, and evals."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from core.trajectory_bank.embeddings import LocalEmbeddings


# Change this whenever the model, dimensions, prompts, or pooling strategy changes.
DEFAULT_EMBEDDING_IDENTITY = "onnx-all-MiniLM-L6-v2:913d7300:mean-220-token-windows:v1"


def get_default_embedding_model() -> Embeddings:
    """Construct the default model without embedding text or downloading weights."""
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return LocalEmbeddings()
