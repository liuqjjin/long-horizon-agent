"""Query-side embedder. Same model as the flow, so vectors are comparable.

Kept inside live_context (it may touch the embedding lib). Used only by the
paper/experiment vector query path.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(text: str, model: str = DEFAULT_MODEL) -> np.ndarray:
    vec = _model(model).encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)
