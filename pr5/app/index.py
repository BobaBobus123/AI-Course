import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings import EMBEDDING_BACKEND, MODEL_NAME, MOCK_MODEL_NAME, embed_texts


INDEX_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_FILE = INDEX_DIR / "vector_index.npy"
METADATA_FILE = INDEX_DIR / "chunks_meta.json"


def save_index(
    embeddings: list[list[float]], chunks: list[dict[str, Any]]
) -> None:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    if vectors.shape[0] != len(chunks):
        raise ValueError("The number of embeddings must match the number of chunks")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain only finite values")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_FILE, vectors)
    metadata = {
        "embedding_backend": EMBEDDING_BACKEND,
        "model_name": MOCK_MODEL_NAME if EMBEDDING_BACKEND == "mock" else MODEL_NAME,
        "chunks": chunks,
    }
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_index() -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not INDEX_FILE.is_file() or not METADATA_FILE.is_file():
        raise FileNotFoundError(
            "Векторний індекс не знайдено. Спочатку запустіть python ingest.py."
        )

    embeddings = np.load(INDEX_FILE)
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    saved_backend = data.get("embedding_backend", "sentence-transformers")
    current_model = MOCK_MODEL_NAME if EMBEDDING_BACKEND == "mock" else MODEL_NAME
    if saved_backend != EMBEDDING_BACKEND or data.get("model_name") != current_model:
        raise ValueError(
            "Режим або модель ембедингів у збереженому індексі не збігається з поточними. "
            "Перебудуйте індекс командою python ingest.py."
        )
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Метадані індексу мають некоректний формат")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Векторний індекс і метадані мають різні розміри")

    return embeddings, chunks


def matches_filters(chunk: dict[str, Any], filters: dict[str, str] | None) -> bool:
    metadata = chunk.get("metadata") or {}
    return all(
        str(metadata.get(key, "")).casefold() == str(value).casefold()
        for key, value in (filters or {}).items()
        if value
    )


def search_vector(
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    embeddings, chunks = load_index()
    if not chunks:
        return []

    query_vec = np.asarray(embed_texts([query], is_query=True)[0], dtype=np.float32)
    if embeddings.shape[1] != query_vec.shape[0]:
        raise ValueError("Поточна модель повернула вектор іншого розміру, ніж індекс")

    similarities = embeddings @ query_vec
    ranked_indices = np.argsort(-similarities, kind="stable")
    results = []
    for index in ranked_indices:
        score = float(similarities[index])
        chunk = chunks[index]
        if score < threshold or not matches_filters(chunk, filters):
            continue
        results.append(
            {
                "score": round(score, 4),
                "text": chunk["text"],
                "source": chunk["source"],
                "metadata": chunk.get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            break

    return results
