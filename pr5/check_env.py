from app.embeddings import EMBEDDING_BACKEND, embed_texts
import numpy as np

print(f"[*] Перевірка ембедінгів (режим: {EMBEDDING_BACKEND})...")
try:
    test_texts = ["Як повернути товар?", "Умови гарантії на смартфони", " livraison colis"]
    vectors = embed_texts(test_texts, is_query=True)
    print(f"[OK] Успішно закодовано тестових рядків: {len(vectors)}")
    print(f" - Розмірність вектора: {len(vectors[0])}")
    if vectors:
        norms = np.linalg.norm(vectors, axis=1)
        print(f" - Норми векторів: {norms.round(4).tolist()}")
    print("[OK] Ембедінги працюють без API-ключа.")
except Exception as e:
    print(f"[X] Помилка завантаження моделі: {e}")