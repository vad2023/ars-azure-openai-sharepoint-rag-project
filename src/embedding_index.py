from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DocumentChunk:
    id: str
    source_file: str
    text: str
    embedding: np.ndarray


class EmbeddingIndex:
    """Simple in-memory vector index using cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: List[DocumentChunk] = []

    def add_chunk(
        self,
        chunk_id: str,
        source_file: str,
        text: str,
        embedding: List[float],
    ) -> None:
        vec = np.array(embedding, dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {vec.shape[0]}"
            )
        self.chunks.append(DocumentChunk(chunk_id, source_file, text, vec))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        mat = np.stack([c.embedding for c in self.chunks])

        # Cosine similarity
        mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        sims = mat_norm @ q_norm

        idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idx]
