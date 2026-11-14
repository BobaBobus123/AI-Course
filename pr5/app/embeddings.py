import hashlib
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence

from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().casefold()
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small"
).strip()
MOCK_MODEL_NAME = "mock-hashing-v1"
MOCK_DIMENSIONS = 384
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_model = None


def get_embedding_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[*] Завантаження локальної моделі ембедінгів: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _mock_embed_texts(texts: Sequence[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * MOCK_DIMENSIONS
        tokens = _TOKEN_RE.findall(text.casefold())
        features = [(token, 1.0) for token in tokens]
        features.extend(
            (f"{left}_{right}", 0.7)
            for left, right in zip(tokens, tokens[1:])
        )

        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big")
            bucket = value % MOCK_DIMENSIONS
            sign = 1.0 if value & (1 << 63) else -1.0
            vector[bucket] += sign * weight

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        vectors.append(vector)
    return vectors


def embed_texts(texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []

    if EMBEDDING_BACKEND == "mock":
        return _mock_embed_texts(texts)
    if EMBEDDING_BACKEND != "sentence-transformers":
        raise ValueError(
            f"Невідомий EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}. "
            "Використайте 'sentence-transformers' або 'mock'."
        )

    model = get_embedding_model()
    if "e5" in MODEL_NAME.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + text for text in texts]

    embeddings = model.encode(list(texts), normalize_embeddings=True)
    return embeddings.tolist()