import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get connection details from environment
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    collection_name = os.getenv("COLLECTION_NAME", "documents")

    # Initialize client
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    try:
        # Check if collection exists
        existing_collections = client.get_collections().collections
        existing_collection_names = [c.name for c in existing_collections]

        if collection_name not in existing_collection_names:
            logger.info(f"Creating collection: {collection_name}")
            
            # Create collection with standard configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Default vector size for common embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    except Exception as e:
        logger.error(f"Error setting up collections: {str(e)}")
        raise

if __name__ == "__main__":
    main()
