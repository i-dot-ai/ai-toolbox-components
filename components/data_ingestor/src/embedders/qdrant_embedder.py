"""
Qdrant embedder using FastEmbed for embedding generation.

Uses qdrant-client's built-in FastEmbed integration so that embedding
is handled automatically - no separate embedding model management needed.
"""

import hashlib
import logging
import os
from typing import Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from parsers.base import ParsedDocument
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantEmbedder(BaseEmbedder):
    """
    Embedder that stores documents in Qdrant using FastEmbed.

    FastEmbed handles embedding generation via ONNX Runtime -
    lightweight, fast, and no PyTorch dependency.

    Connection settings (host/port) come from environment variables.
    Behavioral settings (model_name/batch_size) come from config file.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32
    ):
        # Connection settings from environment
        self.host = os.environ.get("VECTOR_DB_HOST", "localhost")
        self.port = int(os.environ.get("VECTOR_DB_PORT", "6333"))

        # Behavioral settings from config
        self.model_name = model_name
        self.batch_size = batch_size

        self._client: Optional[QdrantClient] = None
        self._embedding_model: Optional[TextEmbedding] = None

        logger.info(f"QdrantEmbedder initialised: host={self.host}, port={self.port}, "
                    f"model={self.model_name}, batch_size={self.batch_size}")

    @property
    def store_type(self) -> str:
        return "qdrant"

    @property
    def client(self) -> QdrantClient:
        """Lazy initialisation of Qdrant client."""
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy initialisation of FastEmbed model."""
        if self._embedding_model is None:
            logger.info(f"Loading FastEmbed model: {self.model_name}")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def _generate_id(self, source: str) -> str:
        """Generate a deterministic ID for a document based on its source."""
        return hashlib.md5(source.encode()).hexdigest()

    def _ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists, creating it if necessary."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            return

        # Get vector size from a test embedding
        test_embedding = list(self.embedding_model.embed(["test"]))[0]
        vector_size = len(test_embedding)

        logger.info(f"Creating collection '{collection_name}' "
                    f"(model: {self.model_name}, dimensions: {vector_size})")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def store(
        self,
        documents: list[ParsedDocument],
        collection_name: str
    ) -> int:
        """Embed and store documents in Qdrant using FastEmbed."""
        if not documents:
            logger.warning("No documents to store")
            return 0

        self._ensure_collection(collection_name)

        texts = [doc.content for doc in documents]

        logger.info(f"Embedding {len(texts)} documents with FastEmbed...")
        embeddings = list(self.embedding_model.embed(texts))

        points = []
        for doc, embedding in zip(documents, embeddings):
            point = PointStruct(
                id=self._generate_id(doc.source),
                vector=embedding.tolist(),
                payload={
                    "source": doc.source,
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp,
                    "source_type": doc.source_type
                }
            )
            points.append(point)

        # Upsert in batches
        stored_count = 0
        total_batches = (len(points) + self.batch_size - 1) // self.batch_size
        for batch_num, i in enumerate(range(0, len(points), self.batch_size), 1):
            batch = points[i:i + self.batch_size]
            logger.info(f"Upserting batch {batch_num}/{total_batches} "
                        f"({len(batch)} points) into '{collection_name}'")
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            stored_count += len(batch)

        logger.info(f"Stored {stored_count}/{len(points)} documents "
                    f"in '{collection_name}'")
        return stored_count
