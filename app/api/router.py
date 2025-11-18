from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
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
        raise HTTPException(status_code=400, detail="Mismatched lengths: embeddings and metadata must have same length")

    if len(req.embeddings) == 0:
        raise HTTPException(status_code=400, detail="Empty embeddings array")

    # Validate dimensions
    for idx, emb in enumerate(req.embeddings):
        if len(emb) != settings.DEFAULT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimension at index {idx}: expected {settings.DEFAULT_DIM}, got {len(emb)}"
            )

    try:
        vectors = [(np.array(vec), meta) for vec, meta in zip(req.embeddings, req.metadata)]
        _store.upsert(vectors)
        return {"count": len(vectors)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert vectors: {str(e)}")

@router.post("/search")
async def search(req: SearchRequest):
    if len(req.query) != settings.DEFAULT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query dimension: expected {settings.DEFAULT_DIM}, got {len(req.query)}"
        )

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")

    if req.k > 1000:
        raise HTTPException(status_code=400, detail="k must not exceed 1000")

    try:
        results = _store.search(np.array(req.query), k=req.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search vectors: {str(e)}")
