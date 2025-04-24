from sentence_transformers import SentenceTransformer
from typing import List
from fastapi import HTTPException
from config.config import zillizconfig
import numpy as np
import logging
from openai import OpenAI, OpenAIError

class EmbeddingService:
    """
    This class provides services for creating and generating passage as well as query embedddings
    """
    def __init__(self):
        model_name = "jinaai/jina-embeddings-v3"
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.task = "retrieval.passage"
        try:
            logging.info("Initializing OpenAI client")
            self.client = None
            self.connect()
            logging.info("OpenAI client initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def _pad_embedding(self, vector: np.ndarray) -> List[float]:
        """
        Pad """
        vector = vector.tolist()
        current_dim = len(vector)
        target_dim = zillizconfig.VECTOR_DIMENSION

        if current_dim > target_dim:
            return vector[:target_dim]
        elif current_dim < target_dim:
            return vector + [0.0] * (target_dim - current_dim)
        return vector

    def connect(self):
        try:
            logging.info("Connecting to OpenAI")
            self.client = OpenAI(api_key=zillizconfig.OPENAI_API_KEY)
            logging.info("Successfully connected to OpenAI")
        except Exception as e:
            logging.error(f"Error connecting to OpenAI: {str(e)}")
            raise HTTPException(status_code=500, detail="Error connecting to OpenAI")
        
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
    
    async def get_openaiembeddings(self, text: List[str]) -> List[float]:
        """
        Create embedddingusing openai model
        """
        try:
            logging.debug(f"Getting embedding for list of texts of (length: {len(text)})")
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            logging.debug("Successfully generated embedding")
            return [item.embedding for item in response.data]
        except OpenAIError as e:
            logging.error(f"Error getting embedding: {str(e)}")
            raise HTTPException(status_code=400, detail="Error getting embedding")
