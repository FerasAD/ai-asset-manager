import numpy as np
from sentence_transformers import SentenceTransformer
from services.audio_embedder import embed_text_query

# Loaded once at import time — only used if CLAP embeddings are not available yet
_text_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_asset_text(filename, tags):
    # Combines filename and tags into a single string for the text fallback path
    tags_text = " ".join(tags) if tags else ""
    return f"{filename} {tags_text}".strip()


def semantic_search_audio(query: str, embedded_assets: list, top_k: int = 20) -> list:
    # Primary semantic search — encodes the query with CLAP and compares against stored audio embeddings
    # Dot product works as cosine similarity because both vectors are unit-normalised
    if not embedded_assets:
        return []

    query_embedding = embed_text_query(query)
    if query_embedding is None:
        return []

    scored = []
    for item in embedded_assets:
        score = float(np.dot(query_embedding, item["embedding"]))
        scored.append({"asset": item["asset"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def semantic_search_text_fallback(query: str, assets_with_texts: list, top_k: int = 20) -> list:
    # Fallback used when no CLAP embeddings exist yet — compares query against filename+tag text
    if not assets_with_texts:
        return []

    from sentence_transformers.util import cos_sim

    asset_texts = [item["text"] for item in assets_with_texts]

    query_embedding = _text_model.encode(query, convert_to_tensor=True)
    asset_embeddings = _text_model.encode(asset_texts, convert_to_tensor=True)
    similarities = cos_sim(query_embedding, asset_embeddings)[0]

    scored = []
    for i, score in enumerate(similarities):
        scored.append({"asset": assets_with_texts[i]["asset"], "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]