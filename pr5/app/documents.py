import os
import re
from pathlib import Path
from typing import List, Dict, Any

def parse_md_file(file_path: Path) -> Dict[str, Any]:
    """Читає файл, розбирає метадані з блоку на початку та текст."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata = {}
    body = content

    # Розбір метаданих у форматі YAML-like frontmatter або заголовків
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            meta_raw = parts[1]
            body = parts[2]
            for line in meta_raw.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip()

    return {
        "filename": file_path.name,
        "metadata": metadata,
        "content": body.strip()
    }

def chunk_text(doc: Dict[str, Any], chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
    """Ділить текст документа на осмислені фрагменти з урахуванням метаданих."""
    text = doc["content"]
    filename = doc["filename"]
    meta = doc["metadata"]
    
    # Розбиваємо за заголовками або абзацами
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += "\n\n" + p if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p
    if current_chunk:
        chunks.append(current_chunk.strip())

    result_chunks = []
    for i, chunk in enumerate(chunks):
        # Додаємо назву документа та заголовок до тексту фрагмента для кращого контексту
        enriched_text = f"Документ: {filename} | Розділ/Частина {i+1}\n{chunk}"
        result_chunks.append({
            "id": f"{filename}_chunk_{i}",
            "text": enriched_text,
            "raw_text": chunk,
            "source": filename,
            "metadata": meta
        })
        
    return result_chunks

def load_all_documents(docs_dir: str) -> List[Dict[str, Any]]:
    docs_path = Path(docs_dir)
    all_chunks = []
    if not docs_path.exists():
        return all_chunks
        
    for file_path in docs_path.glob("*.md"):
        if file_path.name.startswith("README"):
            continue
        doc = parse_md_file(file_path)
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        
    return all_chunks