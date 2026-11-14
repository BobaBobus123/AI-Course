import re
from pathlib import Path
from typing import Any

import yaml


_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)(?:\r?\n)(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_md_file(file_path: Path) -> dict[str, Any]:
    """Read Markdown content and YAML frontmatter from a knowledge-base file."""
    content = file_path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return {"filename": file_path.name, "metadata": {}, "content": content.strip()}

    try:
        metadata = yaml.load(match.group(1), Loader=yaml.BaseLoader) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter in {file_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"YAML frontmatter in {file_path} must be a mapping")

    return {
        "filename": file_path.name,
        "metadata": metadata,
        "content": content[match.end():].strip(),
    }


def _split_long_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def chunk_text(
    doc: dict[str, Any], chunk_size: int = 400, overlap: int = 50
) -> list[dict[str, Any]]:
    """Split document sections into overlapping, context-enriched chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    filename = doc["filename"]
    metadata = doc.get("metadata") or {}
    title = metadata.get("title") or Path(filename).stem
    sections: list[tuple[str, list[str]]] = []
    current_section = "Загальний розділ"
    current_lines: list[str] = []

    for line in doc.get("content", "").splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = heading.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_section, current_lines))

    chunks = []
    chunk_number = 0
    for section, lines in sections:
        section_text = "\n".join(lines).strip()
        if not section_text:
            continue
        for raw_text in _split_long_section(section_text, chunk_size, overlap):
            chunks.append(
                {
                    "id": f"{filename}_chunk_{chunk_number}",
                    "text": f"Документ: {title} | Розділ: {section}\n{raw_text}",
                    "raw_text": raw_text,
                    "source": filename,
                    "section": section,
                    "metadata": metadata,
                }
            )
            chunk_number += 1

    return chunks


def load_all_documents(docs_dir: str | Path) -> list[dict[str, Any]]:
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        return []

    all_chunks = []
    for file_path in sorted(docs_path.rglob("*.md")):
        if file_path.name.casefold() == "readme.md":
            continue
        all_chunks.extend(chunk_text(parse_md_file(file_path)))
    return all_chunks
