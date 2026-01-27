"""
Example plugin for vector_db.

Plugins in this directory are automatically executed after Qdrant starts.
Use plugins for custom initialisation tasks like creating specific collections
or setting up indexes.

This example plugin does nothing - delete or modify it as needed.
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example plugin entry point."""
    logger.info("Example plugin executed - modify or delete this file")

    # Example: You could create a collection with specific settings here
    # from qdrant_client import QdrantClient
    # from qdrant_client.http import models
    #
    # client = QdrantClient(host="localhost", port=6333)
    # client.create_collection(
    #     collection_name="my_collection",
    #     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    # )


if __name__ == "__main__":
    main()
