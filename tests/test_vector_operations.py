from fastapi.testclient import TestClient
from app.main import app
import numpy as np

client = TestClient(app)

# Test data - using DEFAULT_DIM=1536
DIM = 1536


def generate_random_embedding(dim=DIM):
    """Generate a random embedding of specified dimension."""
    return np.random.rand(dim).tolist()


class TestUpsert:
    """Test cases for /v1/upsert endpoint."""

    def test_upsert_success(self):
        """Test successful upsert of valid embeddings."""
        embeddings = [generate_random_embedding() for _ in range(3)]
        metadata = [{"id": i, "text": f"doc_{i}"} for i in range(3)]

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 200
        assert response.json() == {"count": 3}

    def test_upsert_single_vector(self):
        """Test upserting a single vector."""
        embeddings = [generate_random_embedding()]
        metadata = [{"id": 0, "text": "single_doc"}]

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 200
        assert response.json() == {"count": 1}

    def test_upsert_mismatched_lengths(self):
        """Test error when embeddings and metadata have different lengths."""
        embeddings = [generate_random_embedding() for _ in range(3)]
        metadata = [{"id": i} for i in range(2)]  # Only 2 metadata items

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 400
        assert "Mismatched lengths" in response.json()["detail"]

    def test_upsert_empty_array(self):
        """Test error when providing empty embeddings array."""
        response = client.post(
            "/v1/upsert",
            json={"embeddings": [], "metadata": []}
        )

        assert response.status_code == 400
        assert "Empty embeddings" in response.json()["detail"]

    def test_upsert_wrong_dimension(self):
        """Test error when embedding has wrong dimension."""
        # Create embedding with wrong dimension (512 instead of 1536)
        embeddings = [generate_random_embedding(dim=512)]
        metadata = [{"id": 0}]

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 400
        assert "Invalid dimension" in response.json()["detail"]
        assert "expected 1536" in response.json()["detail"]
        assert "got 512" in response.json()["detail"]

    def test_upsert_mixed_dimensions(self):
        """Test error when embeddings have inconsistent dimensions."""
        embeddings = [
            generate_random_embedding(dim=1536),  # Correct
            generate_random_embedding(dim=768),   # Wrong
            generate_random_embedding(dim=1536),  # Correct
        ]
        metadata = [{"id": i} for i in range(3)]

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 400
        assert "Invalid dimension at index 1" in response.json()["detail"]

    def test_upsert_with_complex_metadata(self):
        """Test upsert with complex metadata structures."""
        embeddings = [generate_random_embedding() for _ in range(2)]
        metadata = [
            {
                "id": 1,
                "text": "Complex document",
                "tags": ["ai", "ml", "nlp"],
                "nested": {"author": "Alice", "date": "2025-11-18"}
            },
            {
                "id": 2,
                "text": "Another doc",
                "score": 0.95,
                "metadata": {"category": "research"}
            }
        ]

        response = client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

        assert response.status_code == 200
        assert response.json() == {"count": 2}


class TestSearch:
    """Test cases for /v1/search endpoint."""

    def setup_method(self):
        """Set up test data before each test."""
        # Insert some test vectors
        embeddings = [generate_random_embedding() for _ in range(10)]
        metadata = [{"id": i, "text": f"doc_{i}"} for i in range(10)]

        client.post(
            "/v1/upsert",
            json={"embeddings": embeddings, "metadata": metadata}
        )

    def test_search_success(self):
        """Test successful search with valid query."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query, "k": 5}
        )

        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert len(result["results"]) <= 5
        # Verify result structure
        for item in result["results"]:
            assert "distance" in item
            assert "meta" in item
            assert isinstance(item["distance"], (int, float))

    def test_search_default_k(self):
        """Test search with default k value."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query}  # k defaults to 5
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) <= 5

    def test_search_custom_k(self):
        """Test search with custom k value."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query, "k": 3}
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) <= 3

    def test_search_wrong_dimension(self):
        """Test error when query has wrong dimension."""
        query = generate_random_embedding(dim=768)  # Wrong dimension

        response = client.post(
            "/v1/search",
            json={"query": query, "k": 5}
        )

        assert response.status_code == 400
        assert "Invalid query dimension" in response.json()["detail"]
        assert "expected 1536" in response.json()["detail"]

    def test_search_invalid_k_zero(self):
        """Test error when k is zero."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query, "k": 0}
        )

        assert response.status_code == 400
        assert "k must be greater than 0" in response.json()["detail"]

    def test_search_invalid_k_negative(self):
        """Test error when k is negative."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query, "k": -5}
        )

        assert response.status_code == 400
        assert "k must be greater than 0" in response.json()["detail"]

    def test_search_invalid_k_too_large(self):
        """Test error when k exceeds maximum."""
        query = generate_random_embedding()

        response = client.post(
            "/v1/search",
            json={"query": query, "k": 1001}
        )

        assert response.status_code == 400
        assert "k must not exceed 1000" in response.json()["detail"]

    def test_search_k_larger_than_available(self):
        """Test search when k is larger than available vectors."""
        query = generate_random_embedding()

        # Request more results than available (we only have ~10)
        response = client.post(
            "/v1/search",
            json={"query": query, "k": 50}
        )

        assert response.status_code == 200
        result = response.json()
        # Should return all available vectors (not fail)
        assert len(result["results"]) <= 50


class TestIntegration:
    """Integration tests for upsert and search workflow."""

    def test_upsert_then_search_exact_match(self):
        """Test that searching for an upserted vector returns it first."""
        # Create a specific embedding
        embedding = generate_random_embedding()
        metadata = [{"id": 999, "text": "exact_match_test"}]

        # Upsert it
        upsert_response = client.post(
            "/v1/upsert",
            json={"embeddings": [embedding], "metadata": metadata}
        )
        assert upsert_response.status_code == 200

        # Search for the exact same embedding
        search_response = client.post(
            "/v1/search",
            json={"query": embedding, "k": 1}
        )
        assert search_response.status_code == 200

        results = search_response.json()["results"]
        assert len(results) > 0
        # First result should be our exact match with distance ~0
        assert results[0]["distance"] < 0.001  # Very small distance for exact match
        assert results[0]["meta"]["id"] == 999

    def test_multiple_upserts_and_search(self):
        """Test multiple upsert operations followed by search."""
        # First batch
        batch1 = [generate_random_embedding() for _ in range(5)]
        meta1 = [{"batch": 1, "id": i} for i in range(5)]

        response1 = client.post(
            "/v1/upsert",
            json={"embeddings": batch1, "metadata": meta1}
        )
        assert response1.status_code == 200

        # Second batch
        batch2 = [generate_random_embedding() for _ in range(3)]
        meta2 = [{"batch": 2, "id": i} for i in range(3)]

        response2 = client.post(
            "/v1/upsert",
            json={"embeddings": batch2, "metadata": meta2}
        )
        assert response2.status_code == 200

        # Search for one of the vectors we just inserted (from batch1)
        # This should find it as the top result
        search_response = client.post(
            "/v1/search",
            json={"query": batch1[0], "k": 10}
        )

        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert len(results) > 0  # Should have results

        # The first result should be from our batch (near-exact match)
        assert results[0]["distance"] < 0.001
        assert "meta" in results[0]
        assert "batch" in results[0]["meta"]
        assert results[0]["meta"]["batch"] in [1, 2]
