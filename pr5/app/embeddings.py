import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[*] Завантаження локальної моделі ембедінгів: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = get_embedding_model()
    # Для моделі E5 рекомендується додавати префікси "query: " або "passage: "
    if "e5" in MODEL_NAME.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()