"""
Base classes for embedders.
"""

from abc import ABC, abstractmethod

from parsers.base import ParsedDocument


class BaseEmbedder(ABC):
    """Abstract base class for vector store embedders."""

    @property
    @abstractmethod
    def store_type(self) -> str:
        """Return the store type identifier."""
        ...

    @abstractmethod
    def store(self, documents: list[ParsedDocument], collection_name: str) -> int:
        """
        Embed and store documents.

        Args:
            documents: List of parsed documents to embed and store
            collection_name: Name of the collection to store in

        Returns:
            Number of documents successfully stored
        """
        ...
