import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.index import load_index

def tokenize(text: str) -> List[str]:
    # Простий токенізатор за словами
    return re.findall(r'\w+', text.lower())

def search_bm25(query: str, top_k: int = 5, filters: dict = None) -> List[Dict[str, Any]]:
    try:
        _, chunks = load_index()
    except FileNotFoundError:
        return []

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    results = []
    for idx in sorted_indices:
        score = float(scores[idx])
        if score <= 0:
            break
            
        chunk = chunks[idx]
        
        # Фільтри за метаданими
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