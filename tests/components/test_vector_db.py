"""
Integration tests for the vector_db component.

Requires Docker to build and run the vector_db container.
"""

import pytest
import requests

from tests.test_utils import verify_service_health


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_health_endpoint(component_endpoint):
    """Test that vector_db starts and becomes healthy."""
    return verify_service_health("vector_db", timeout=120)


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_create_collection(component_endpoint):
    """Test we can create a new collection via Qdrant HTTP API."""
    collection_name = "test-collection"
    payload = {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }
    response = requests.put(
        f"{component_endpoint}/collections/{collection_name}",
        json=payload,
    )
    assert response.status_code == 200


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_get_collections(component_endpoint):
    """Test we can retrieve existing collections."""
    response = requests.get(f"{component_endpoint}/collections")
    assert response.status_code == 200
    assert "collections" in response.json()["result"]


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_delete_collection(component_endpoint):
    """Test we can delete an existing collection."""
    collection_name = "test-collection"
    response = requests.delete(f"{component_endpoint}/collections/{collection_name}")
    assert response.status_code == 200
