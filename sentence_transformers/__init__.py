import hashlib

import numpy as np


class SentenceTransformer:
    """Minimal stub of SentenceTransformer for offline testing."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._dim = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # repeat or truncate to self._dim
            if arr.size < self._dim:
                reps = (self._dim + arr.size - 1) // arr.size
                arr = np.tile(arr, reps)[: self._dim]
            else:
                arr = arr[: self._dim]
            if np.linalg.norm(arr) != 0:
                arr = arr / np.linalg.norm(arr)
            vectors.append(arr)
        return np.vstack(vectors)
