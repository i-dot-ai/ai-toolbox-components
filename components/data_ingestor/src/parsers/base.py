"""
Base parser interface for data ingestion.

All parsers must inherit from BaseParser and implement the parse() method
to produce a standardised ParsedDocument output suitable for embedding.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """
    Standardised document format for embedding pipelines.

    All parsers must produce documents in this format to ensure
    compatibility with downstream embedding and vector storage.
    """
    source: str          # Source identifier (URL, file path, etc.)
    title: str           # Document title
    content: str         # Main text content for embedding
    metadata: dict       # Additional metadata (source-specific)
    timestamp: str       # ISO format timestamp of parsing
    source_type: str     # Type identifier (html, pdf, markdown, etc.)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


class BaseParser(ABC):
    """
    Abstract base class for content parsers.

    Subclasses must implement:
    - fetch(): Retrieve content from a source (default returns None)
    - parse(): Convert raw content to ParsedDocument
    - source_type property: Return the parser's source type identifier
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier (e.g., 'html', 'pdf', 'markdown')."""
        pass

    @abstractmethod
    def parse(self, content: Any, source: str) -> ParsedDocument:
        """
        Parse raw content into a standardised document.

        Args:
            content: Raw content to parse (type depends on parser)
            source: Source identifier (URL, file path, etc.)

        Returns:
            ParsedDocument with extracted content and metadata
        """
        pass

    @abstractmethod
    def fetch(self, source: str) -> Any:
        """
        Fetch content from a source.
        Override this method for parsers that can retrieve their own content.

        Args:
            source: Source identifier to fetch from

        Returns:
            Raw content
        """
        pass

    def ingest(self, source: str) -> Optional[ParsedDocument]:
        """
        Fetch and parse content from a source.

        Args:
            source: Source identifier

        Returns:
            ParsedDocument or None if content could not be fetched
        """
        content = self.fetch(source)
        if content is None:
            logger.warning(f"No content fetched from {source}")
            return None
        return self.parse(content, source)

    @staticmethod
    def _current_timestamp() -> str:
        """Return current time as ISO format string."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
