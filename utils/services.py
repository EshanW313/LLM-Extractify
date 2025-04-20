from sentence_transformers import SentenceTransformer
from typing import List
from fastapi import HTTPException
from config.config import zillizconfig
import numpy as np

class EmbeddingService:
    def __init__(self):
        model_name = "jinaai/jina-embeddings-v3"
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.task = "retrieval.passage"

    def _pad_embedding(self, vector: np.ndarray) -> List[float]:
        vector = vector.tolist()
        current_dim = len(vector)
        target_dim = zillizconfig.VECTOR_DIMENSION

        if current_dim > target_dim:
            return vector[:target_dim]
        elif current_dim < target_dim:
            return vector + [0.0] * (target_dim - current_dim)
        return vector

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                task=self.task,
                prompt_name=self.task,
                normalize_embeddings=True
            )
            return [self._pad_embedding(vec) for vec in embeddings]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Embedding error: {str(e)}")

    async def get_query_embeddings(self, query: str) -> List[float]:
        try:
            embedding = self.model.encode(
                query,
                task="retrieval.query",
                prompt_name="retrieval.query",
                normalize_embeddings=True
            )
            return [self._pad_embedding(embedding)]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Embedding error: {str(e)}")
