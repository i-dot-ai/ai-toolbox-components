import pdb
import pytest
import requests

from tests.test_utils import verify_service_health


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_health_endpoint(component_endpoint):
    return verify_service_health("vector_db", timeout=120)


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_create_collection(component_endpoint):
    """Test we can create a new collection"""
    collection_name = "test-collection"

    # Proper Qdrant collection configuration
    payload = {
        "vectors": {
            "images": {
                "size": 384,  # Must match your embed model dimensions
                "distance": "Cosine"
            }
        }
    }
    response = requests.put(
        f"{component_endpoint}/collections/{collection_name}",
        data=str(payload).replace("'", '"')
    )
    assert response.status_code == 200


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_get_collections(component_endpoint):
    """Test we can retrieve existing collections"""
    response = requests.get(f"{component_endpoint}/collections")
    assert response.status_code == 200
    assert "collections" in response.json()['result']


@pytest.mark.parametrize("component_endpoint", [("vector_db", "6333")], indirect=True)
def test_delete_collection(component_endpoint):
    """Test we can delete an existing collection"""
    collection_name = "test-collection"
    response = requests.delete(f"{component_endpoint}/collections/{collection_name}")
    assert response.status_code == 200
