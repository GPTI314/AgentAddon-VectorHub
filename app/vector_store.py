from typing import List, Tuple
import faiss, numpy as np

class InMemoryFaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self._vectors: List[np.ndarray] = []
        self._meta: List[dict] = []

    def upsert(self, embeddings: List[Tuple[np.ndarray, dict]]):
        arr = np.vstack([e[0] for e in embeddings]).astype('float32')
        self.index.add(arr)
        for emb, meta in embeddings:
            self._vectors.append(emb)
            self._meta.append(meta)

    def search(self, query: np.ndarray, k: int = 5):
        query = query.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i < len(self._meta):
                results.append({"distance": float(d), "meta": self._meta[i]})
        return results
