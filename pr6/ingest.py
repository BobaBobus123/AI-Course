from pathlib import Path
from app.documents import load_all_documents
from app.embeddings import embed_texts
from app.index import save_index

def main():
    docs_dir = Path(__file__).resolve().parent / "docs"
    print(f"[*] Читання документів із папки: {docs_dir}")
    
    chunks = load_all_documents(str(docs_dir))
    print(f"[*] Загалом сформовано фрагментів: {len(chunks)}")
    
    if not chunks:
        print("[!] Документів не знайдено!")
        return
        
    texts = [c["text"] for c in chunks]
    print("[*] Генерація ембедінгів для фрагментів...")
    embeddings = embed_texts(texts, is_query=False)
    
    print("[*] З збереження індексу на диск...")
    save_index(embeddings, chunks)
    print("[OK] Індекс успішно побудовано та збережено!")

if __name__ == "__main__":
    main()