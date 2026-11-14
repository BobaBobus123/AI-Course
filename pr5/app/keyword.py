import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.index import load_index, matches_filters


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "with",
    "а", "але", "без", "би", "був", "була", "було", "були", "в", "від",
    "до", "для", "з", "за", "і", "й", "на", "над", "нехай", "чи", "що",
    "як", "у", "та", "це", "цей", "ця", "ці", "із",
    "без", "бы", "в", "во", "для", "до", "за", "и", "из", "к", "как",
    "на", "над", "но", "о", "об", "от", "по", "с", "со", "так", "то",
    "у", "что", "это",
}
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize Ukrainian, Russian, and Latin text consistently for BM25."""
    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if token not in _STOP_WORDS
    ]


def search_bm25(
    query: str, top_k: int = 5, filters: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    _, all_chunks = load_index()
    chunks = [chunk for chunk in all_chunks if matches_filters(chunk, filters)]
    if not chunks:
        return []

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            break
        chunk = chunks[index]
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
