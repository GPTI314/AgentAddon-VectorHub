from fastapi import APIRouter
from pydantic import BaseModel
from ..config import get_settings
from ..vector_store import InMemoryFaissStore
import numpy as np

settings = get_settings()
_store = InMemoryFaissStore(dim=settings.DEFAULT_DIM)
router = APIRouter(prefix="/v1")

class UpsertRequest(BaseModel):
    embeddings: list[list[float]]
    metadata: list[dict]

class SearchRequest(BaseModel):
    query: list[float]
    k: int = 5

@router.post("/upsert")
async def upsert(req: UpsertRequest):
    if len(req.embeddings) != len(req.metadata):
        return {"error": "Mismatched lengths"}
    vectors = [(np.array(vec), meta) for vec, meta in zip(req.embeddings, req.metadata)]
    _store.upsert(vectors)
    return {"count": len(vectors)}

@router.post("/search")
async def search(req: SearchRequest):
    results = _store.search(np.array(req.query), k=req.k)
    return {"results": results}
