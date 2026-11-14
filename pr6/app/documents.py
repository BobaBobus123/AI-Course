import re
from pathlib import Path
from typing import Any


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_md_file(file_path: Path) -> dict[str, Any]:
    """Read Markdown content and its simple YAML-style frontmatter."""
    content = file_path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

    return {
        "filename": file_path.name,
        "metadata": metadata,
        "content": body.strip(),
    }


def _split_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end
        if target_end < len(text):
            minimum_end = start + chunk_size // 2
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, minimum_end, target_end)
                if boundary > start:
                    end = boundary
                    break

        passage = text[start:end].strip()
        if passage:
            chunks.append(passage)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def _contains_body_text(section_text: str) -> bool:
    without_headings = _HEADING_RE.sub("", section_text)
    return bool(without_headings.strip())


def chunk_text(
    doc: dict[str, Any], chunk_size: int = 400, overlap: int = 50
) -> list[dict[str, Any]]:
    """Split Markdown at heading boundaries and retain each chunk's real section."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    filename = doc["filename"]
    metadata = doc.get("metadata") or {}
    title = metadata.get("title") or Path(filename).stem
    heading_stack: list[str] = []
    current_section = "Загальний розділ"
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    for line in doc.get("content", "").splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text and _contains_body_text(section_text):
                    sections.append((current_section, section_text))

            level = len(heading.group(1))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading.group(2).strip())
            current_section = " / ".join(heading_stack)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text and _contains_body_text(section_text):
            sections.append((current_section, section_text))

    chunks = []
    chunk_number = 0
    for section, section_text in sections:
        for raw_text in _split_section(section_text, chunk_size, overlap):
            chunks.append(
                {
                    "id": f"{filename}_chunk_{chunk_number}",
                    "text": (
                        f"Документ: {title} | Розділ: {section}\n"
                        f"{raw_text}"
                    ),
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
