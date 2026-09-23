from app.embeddings import embed_texts
import numpy as np

print("[*] Перевірка роботи локальної моделі ембедінгів...")
try:
    test_texts = ["Як повернути товар?", "Умови гарантії на смартфони", " livraison colis"]
    vectors = embed_texts(test_texts, is_query=True)
    print(f"[OK] Успішно закодовано тестових рядків: {len(vectors)}")
    print(f" - Розмірність вектора: {len(vectors[0])}")
    print("[OK] Локальна модель працює коректно без підключення до інтернету чи API-ключів!")
except Exception as e:
    print(f"[X] Помилка завантаження моделі: {e}")