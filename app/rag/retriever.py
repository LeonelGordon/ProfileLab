from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleRetriever:
    def __init__(self, chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        if not chunks:
            raise ValueError("La lista de chunks no puede estar vacía")

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = self.model.encode(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not query.strip():
            raise ValueError("La query no puede estar vacía")

        if top_k <= 0:
            raise ValueError("top_k debe ser mayor a 0")

        query_embedding = self.model.encode(query)

        similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]

    @staticmethod
    def _cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

        if query_norm == 0:
            raise ValueError("El embedding de la query no es válido")

        if np.any(chunk_norms == 0):
            raise ValueError("Uno o más embeddings de chunks no son válidos")

        similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)

        return similarities