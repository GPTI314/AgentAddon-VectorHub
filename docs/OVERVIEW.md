# VectorHub Overview

VectorHub offers a pluggable embedding storage and retrieval API for AI agent memory augmentation.

## Responsibilities
- Store embeddings with associated metadata
- Provide similarity search (initial FAISS backend)
- Abstract interface for future backends (PgVector, Milvus, Qdrant)

## API Summary
- POST /v1/upsert: batch upsert embeddings + metadata
- POST /v1/search: vector similarity query
- GET /health: service status

## Data Contract
```
UpsertRequest {
  embeddings: number[][]   // shape: N x dim
  metadata: object[]       // length N, arbitrary JSON
}
SearchRequest {
  query: number[]          // length dim
  k: number                // result count
}
```

## Extensibility
- Adapter registry for multiple vector stores
- Hybrid search (vector + keyword filters) planned
- Namespaces / multi-tenancy design TBD

## Future Enhancements
- Persistent storage layer
- Embedding versioning & soft deletes
- Metrics & monitoring (queries/sec, latency)
