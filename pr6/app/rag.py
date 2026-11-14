from time import perf_counter
from typing import Any

from app.llm import NO_INFORMATION_TEXT, generate_rag_response
from app.retrieval import retrieve_and_filter_chunks


def run_rag(
    query: str, filters: dict[str, str] | None = None
) -> dict[str, Any]:
    search_started = perf_counter()
    chunks = retrieve_and_filter_chunks(query, filters=filters)
    search_time_ms = round((perf_counter() - search_started) * 1000, 2)

    if not chunks:
        return {
            "query": query,
            "status": "у базі немає",
            "response_text": NO_INFORMATION_TEXT,
            "found_in_base": False,
            "sources": [],
            "search_time_ms": search_time_ms,
            "generation_time_ms": 0.0,
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "context_chunks": [],
            "model": "none",
        }

    generation_started = perf_counter()
    result = generate_rag_response(query, chunks)
    generation_time_ms = round((perf_counter() - generation_started) * 1000, 2)
    found_in_base = bool(result["found_in_base"] and result["sources"])

    return {
        "query": query,
        "status": "за базою знань" if found_in_base else "у базі немає",
        "response_text": result["response_text"],
        "found_in_base": found_in_base,
        "sources": result["sources"] if found_in_base else [],
        "search_time_ms": search_time_ms,
        "generation_time_ms": generation_time_ms,
        "tokens": result["tokens"],
        "context_chunks": result["raw_chunks"],
        "model": result["model"],
    }
