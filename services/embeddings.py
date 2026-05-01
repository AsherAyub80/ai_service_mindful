# ai_service/services/embeddings.py
"""
Local sentence-transformer embeddings.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: all-MiniLM-L6-v2
  - 22MB download, runs on CPU
  - 384 dimensions
  - 100% FREE — no API key, no rate limits, no cost ever
  - Fast: ~10ms per embedding on CPU

This is why we have a Python service:
Node.js cannot run sentence-transformers natively.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import asyncio


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"

    async def load(self):
        """Load model in a thread so we don't block the event loop."""
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(self.model_name)
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single text string → list of 384 floats."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded yet")
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once (faster than one-by-one)."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded yet")
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b))  # vectors are already L2-normalized

    def rank_by_similarity(
        self,
        query: str,
        candidates: List[dict],
        text_field: str = "description",
        top_k: int = 10,
    ) -> List[dict]:
        """
        Given a query string and a list of dicts,
        rank them by semantic similarity to the query.
        """
        if not candidates:
            return []

        query_vec = np.array(self.embed(query))
        texts = [c.get(text_field, c.get("title", "")) for c in candidates]
        candidate_vecs = np.array(self.embed_batch(texts))

        # Batch cosine similarity
        scores = candidate_vecs @ query_vec  # dot product (normalized = cosine)

        # Attach scores and sort
        for i, c in enumerate(candidates):
            c["_similarity_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["_similarity_score"], reverse=True)
        return ranked[:top_k]
