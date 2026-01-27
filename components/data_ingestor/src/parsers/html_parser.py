"""
HTML parser for web page content ingestion.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """
    Parser for HTML web pages.

    Fetches HTML from URLs, extracts meaningful text content,
    and produces standardised documents for embedding.
    """

    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (compatible; DataIngestor/1.0)",
        timeout: int = 30,
        exclude_elements: Optional[list[str]] = None
    ):
        """
        Initialise the HTML parser.

        Args:
            user_agent: User agent string for HTTP requests
            timeout: Request timeout in seconds
            exclude_elements: HTML elements to remove before extraction
        """
        self.timeout = timeout
        self.exclude_elements = exclude_elements or [
            "script", "style", "nav", "footer", "header", "aside"
        ]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    @property
    def source_type(self) -> str:
        return "html"

    def fetch(self, source: str) -> Optional[str]:
        """
        Fetch HTML content from a URL.

        Args:
            source: URL to fetch

        Returns:
            HTML string or None on failure
        """
        logger.info(f"Fetching {source}")
        try:
            response = self.session.get(source, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Fetched {source} (status {response.status_code}, "
                        f"{len(response.text)} chars)")
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source}: {e}")
            return None

    def parse(self, content: str, source: str) -> ParsedDocument:
        """
        Parse HTML content into a standardised document.

        Args:
            content: Raw HTML string
            source: Source URL

        Returns:
            ParsedDocument with extracted content
        """
        soup = BeautifulSoup(content, "html.parser")

        # Remove unwanted elements
        for element in soup(self.exclude_elements):
            element.decompose()

        title = self._extract_title(soup)
        text_content = self._extract_content(soup)
        metadata = self._extract_metadata(soup, source)

        logger.info(f"Parsed {source}: title='{title}', "
                    f"content_length={len(text_content)}, "
                    f"metadata_keys={list(metadata.keys())}")

        return ParsedDocument(
            source=source,
            title=title,
            content=text_content,
            metadata=metadata,
            timestamp=self._current_timestamp(),
            source_type=self.source_type
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract document title from HTML."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML."""
        # Try to find main content area
        main_content = soup.find("main") or soup.find("article") or soup.body
        if main_content:
            return main_content.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)

    def _extract_metadata(self, soup: BeautifulSoup, source: str) -> dict:
        """Extract metadata from HTML and URL."""
        parsed_url = urlparse(source)
        metadata = {
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
        }

        # Get meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]

        # Get meta keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords and meta_keywords.get("content"):
            metadata["keywords"] = meta_keywords["content"]

        # Get Open Graph title if different from page title
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            metadata["og_title"] = og_title["content"]

        return metadata
