from __future__ import annotations


class LocalEmbeddings:
    """Explicit local model; pool token windows rather than truncate long memories."""

    identity = "onnx-all-MiniLM-L6-v2:913d7300:mean-220-token-windows:v1"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        from tokenizers import Tokenizer

        if self._model is None:
            self._model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            self._model(["initialize"])
            self._tokenizer = Tokenizer.from_str(self._model.tokenizer.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        vectors = []
        for text in texts:
            offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
            chunks = [text[offsets[i][0]:offsets[min(i + 219, len(offsets) - 1)][1]]
                      for i in range(0, len(offsets), 220)] or [text]
            vector = np.mean(self._model(chunks), axis=0)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Invalid trajectory embedding")
            vectors.append((vector / norm).tolist())
        return vectors
