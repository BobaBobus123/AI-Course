import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from app.index import search_vector

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_N = int(os.getenv("TOP_N_CONTEXT", "4"))
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

def retrieve_and_filter_chunks(query: str, filters: dict = None) -> List[Dict[str, Any]]:
    """Широкий пошук з подальшим жорстким відбором у коді (запобіжники)."""
    print(f"\n[DEBUG] Отримано запит: '{query}' з порогом {THRESHOLD}")
    
    # 1. Пошук через вектори
    raw_results = search_vector(query, top_k=TOP_K, threshold=THRESHOLD, filters=filters)
    print(f"[DEBUG] Знайдено сирих результатів від пошуку: {len(raw_results)}")
    
    filtered_chunks = []
    for chunk in raw_results:
        meta = chunk.get("metadata", {})
        print(f"  - Знайдено фрагмент: джерело={chunk.get('source')}, оцінка={chunk.get('score')}, аудиторія={meta.get('audience')}, статус={meta.get('status')}")
        
        # ЗАПОБІЖНИК КОДУ: Внутрішні документи ніколи не даємо клієнту
        if meta.get("audience") == "персонал" or meta.get("audience") == "оператор":
            print("    [X] Відкинуто внутрішній документ (аудиторія)")
            continue
            
        # Запобіжник для архівних редакцій
        if meta.get("status") == "архівний" or meta.get("status") == "архівна":
            print("    [X] Відкинуто архівний документ")
            continue
            
        filtered_chunks.append(chunk)
        
    # Згортання дублікатів та обмеження кількості фрагментів
    seen_sources = set()
    final_chunks = []
    
    for chunk in filtered_chunks:
        source = chunk["source"]
        if source not in seen_sources:
            seen_sources.add(source)
            final_chunks.append(chunk)
            if len(final_chunks) >= TOP_N:
                break
                
    print(f"[DEBUG] Передано фрагментів моделі після фільтрації: {len(final_chunks)}")
    return final_chunks