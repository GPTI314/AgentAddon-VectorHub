# VectorHub Extended Documentation

## Purpose
Embedding storage & semantic retrieval microservice with adapter-friendly architecture.

## Initial Implementation
- In-memory FAISS index (L2 similarity)
- Upsert endpoint `/v1/upsert`
- Search endpoint `/v1/search`

## Upsert Payload Example
```json
{
  "embeddings": [[0.1,0.2,0.3,...]],
  "metadata": [{"doc_id": "123", "tags": ["example"]}]
}
```

## Future Backend Adapters
- PgVector
- Qdrant
- Milvus
- Redis Vector similarity (experimental)

## Roadmap Prompt Sequence
1. Abstract store interface; adapter registration & selection.
2. Add batch upsert streaming mode.
3. Implement hybrid search (vector + metadata filter).
4. Support deletion & tombstoning.
5. Distance normalization & score calibration.
6. Add authorization (API keys / role-based).
7. Introduce replication & persistence snapshotting.
8. Add memory compaction strategy.
9. Add metrics & Prometheus exporter.
10. Performance profiling & index optimization.

## Security Considerations
- Validate embedding dimensionality.
- Enforce max batch sizes.
- Prevent metadata injection (sanitize keys).

## Configuration (env)
- `DEFAULT_DIM` embedding dimension fallback.
- `LOG_JSON` toggle.

## License
MIT (to be added).
