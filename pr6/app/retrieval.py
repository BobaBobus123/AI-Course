import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv

from app.index import search_vector


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_N = int(os.getenv("TOP_N_CONTEXT", "4"))
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

if TOP_K <= 0 or TOP_N <= 0 or MAX_CONTEXT_TOKENS <= 0:
    raise ValueError("Retrieval counts and MAX_CONTEXT_TOKENS must be positive")
if not -1.0 <= THRESHOLD <= 1.0:
    raise ValueError("SIMILARITY_THRESHOLD must be between -1 and 1")

try:
    _TOKEN_ENCODER = tiktoken.encoding_for_model(LLM_MODEL_NAME)
except KeyError:
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_ARCHIVE_VALUES = {"архів", "архівний", "архівна", "архівне", "archived", "archive"}
_CUSTOMER_AUDIENCES = {
    "клієнт",
    "клієнти",
    "клієнтка",
    "client",
    "clients",
    "customer",
    "customers",
    "public",
}
_PROTECTED_MARKERS = (
    "персонал",
    "оператор",
    "внутрішн",
    "internal",
    "staff",
    "operator",
)


def _normalized(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def _is_internal_document(chunk: dict[str, Any]) -> bool:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return True
    audience = _normalized(metadata.get("audience"))
    category = _normalized(metadata.get("category"))
    source = _normalized(chunk.get("source"))
    content = _normalized(chunk.get("raw_text", chunk.get("text", "")))

    return (
        audience not in _CUSTOMER_AUDIENCES
        or any(marker in audience for marker in _PROTECTED_MARKERS)
        or any(marker in category for marker in _PROTECTED_MARKERS)
        or any(marker in source for marker in _PROTECTED_MARKERS)
        or "внутрішній документ. клієнтам не надається" in content
    )


def _is_archived(chunk: dict[str, Any]) -> bool:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return True
    status = _normalized(metadata.get("status"))
    return status in _ARCHIVE_VALUES or status.startswith("архів")


def _get_section(chunk: dict[str, Any]) -> str:
    section = chunk.get("section")
    if section:
        return str(section)

    text = str(chunk.get("raw_text") or chunk.get("text") or "")
    headings = _HEADING_RE.findall(text)
    if headings:
        return headings[-1].strip()

    header = text.splitlines()[0] if text else ""
    marker = re.search(r"Розділ(?:/Частина)?\s*:\s*(.+)", header, re.IGNORECASE)
    return marker.group(1).strip() if marker else "Основний текст"


def _context_block(chunk: dict[str, Any], source_id: int, passage: str) -> str:
    metadata = chunk.get("metadata") or {}
    document_name = str(metadata.get("title") or chunk.get("source") or "Документ")
    section = _get_section(chunk)
    updated = str(metadata.get("updated") or "не вказано")
    return (
        f"[{source_id}] Документ: {document_name} | "
        f"Розділ: {section} | Редакція: {updated}\n"
        f"Фрагмент:\n{passage}"
    )


def _fit_context_block(
    chunk: dict[str, Any],
    source_id: int,
    max_tokens: int,
    context_prefix: str = "",
) -> tuple[str, str, int] | None:
    passage = str(chunk.get("raw_text") or chunk.get("text") or "").strip()
    if not passage:
        return None

    full_block = _context_block(chunk, source_id, passage)
    full_context = context_prefix + full_block
    full_context_tokens = len(_TOKEN_ENCODER.encode(full_context))
    if full_context_tokens <= max_tokens:
        return full_block, passage, full_context_tokens

    encoded_passage = _TOKEN_ENCODER.encode(passage)
    low, high = 0, len(encoded_passage)
    best: tuple[str, str, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        excerpt = (
            _TOKEN_ENCODER.decode_bytes(encoded_passage[:middle])
            .decode("utf-8", errors="ignore")
            .strip()
        )
        block = _context_block(chunk, source_id, excerpt)
        token_count = len(_TOKEN_ENCODER.encode(context_prefix + block))
        if excerpt and token_count <= max_tokens:
            best = (block, excerpt, token_count)
            low = middle + 1
        else:
            high = middle - 1
    return best


def retrieve_and_filter_chunks(
    query: str, filters: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Retrieve customer-safe chunks and fit numbered excerpts to the context budget."""
    if not query.strip():
        return []

    raw_results = search_vector(
        query,
        top_k=TOP_K,
        threshold=-1.0,
        filters=filters,
    )
    selected: list[dict[str, Any]] = []
    context_blocks: list[str] = []

    for candidate in raw_results:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("metadata"), dict):
            continue
        try:
            score = float(candidate.get("score", float("-inf")))
        except (TypeError, ValueError):
            continue
        if score < THRESHOLD:
            continue
        if _is_internal_document(candidate) or _is_archived(candidate):
            continue

        source_id = len(selected) + 1
        context_prefix = "\n\n---\n\n".join(context_blocks)
        if context_prefix:
            context_prefix += "\n\n---\n\n"
        fitted = _fit_context_block(
            candidate,
            source_id,
            MAX_CONTEXT_TOKENS,
            context_prefix,
        )
        if fitted is None:
            continue

        context_block, passage, token_count = fitted
        chunk = dict(candidate)
        chunk.update(
            {
                "context_id": source_id,
                "section": _get_section(candidate),
                "context_text": passage,
                "context_block": context_block,
                "context_token_count": len(_TOKEN_ENCODER.encode(context_block)),
            }
        )
        selected.append(chunk)
        context_blocks.append(context_block)
        if len(selected) >= TOP_N:
            break

    return selected
