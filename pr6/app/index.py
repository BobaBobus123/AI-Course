import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.embeddings import embed_texts, MODEL_NAME

INDEX_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_FILE = INDEX_DIR / "vector_index.npy"
METADATA_FILE = INDEX_DIR / "chunks_meta.json"

def save_index(embeddings: List[List[float]], chunks: List[Dict[str, Any]]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_FILE, np.array(embeddings, dtype=np.float32))
    
    meta_data = {
        "model_name": MODEL_NAME,
        "chunks": chunks
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Векторний індекс не знайдено. Спочатку запустіть ingest.py!")
    
    embeddings = np.load(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if data.get("model_name") != MODEL_NAME:
        raise ValueError("Назва моделі в індексі не збігається з поточною конфігурацією! Потрібно перебудувати індекс.")
        
    return embeddings, data["chunks"]

def search_vector(query: str, top_k: int = 5, threshold: float = 0.0, filters: dict = None) -> List[Dict[str, Any]]:
    try:
        embeddings, chunks = load_index()
    except FileNotFoundError:
        return []

    query_vec = np.array(embed_texts([query], is_query=True)[0], dtype=np.float32)
    
    # Оскільки вектори нормалізовані, косинусна схожість — це скалярний добуток
    similarities = np.dot(embeddings, query_vec)
    
    # Сортування за спаданням схожості
    sorted_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in sorted_indices:
        score = float(similarities[idx])
        if score < threshold:
            continue
            
        chunk = chunks[idx]
        
        # Застосування фільтрів за метаданими (аудиторія, статус, категорія тощо)
        match_filters = True
        if filters:
            for k, v in filters.items():
                if v and chunk["metadata"].get(k) != v:
                    match_filters = False
                    break
        if not match_filters:
            continue
            
        results.append({
            "score": round(score, 4),
            "text": chunk["text"],
            "source": chunk["source"],
            "metadata": chunk["metadata"]
        })
        
        if len(results) >= top_k:
            break
            
    return results