"""
Unit tests for the data_ingestor component.

Tests parsers, embedders, and the ingestor orchestrator using mocks
so no Docker or external services are needed.
"""

import sys
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add data_ingestor src to path so we can import directly
_src = str(Path(__file__).resolve().parents[2] / "components" / "data_ingestor" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Mock heavy dependencies that aren't installed locally (fastembed, qdrant_client).
# These must be set before importing the embedders package.
_mock_fastembed = ModuleType("fastembed")
_mock_fastembed.TextEmbedding = MagicMock
sys.modules.setdefault("fastembed", _mock_fastembed)

_mock_qdrant = ModuleType("qdrant_client")
_mock_qdrant.QdrantClient = MagicMock
sys.modules.setdefault("qdrant_client", _mock_qdrant)

_mock_qdrant_models = ModuleType("qdrant_client.models")
_mock_qdrant_models.Distance = MagicMock()
_mock_qdrant_models.Distance.COSINE = "Cosine"
_mock_qdrant_models.VectorParams = MagicMock
_mock_qdrant_models.PointStruct = MagicMock
sys.modules.setdefault("qdrant_client.models", _mock_qdrant_models)

from parsers.base import ParsedDocument
from parsers.html_parser import HTMLParser


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<head><title>Test Page</title>
<meta name="description" content="A test page">
<meta name="keywords" content="test,unit">
<meta property="og:title" content="OG Test">
</head>
<body>
<nav>Navigation</nav>
<header>Header</header>
<main>
<h1>Main Heading</h1>
<p>Main content paragraph.</p>
</main>
<footer>Footer</footer>
<script>alert('x')</script>
<style>body{}</style>
</body>
</html>
"""


def _make_doc(source="https://example.com", content="test content"):
    return ParsedDocument(
        source=source, title="Test", content=content,
        metadata={}, timestamp="2025-01-01T00:00:00Z", source_type="html",
    )


# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------

class TestHTMLParser:
    def test_parse_extracts_all_fields(self):
        """Parse sample HTML and verify title, content, metadata, and excluded elements."""
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com/page")

        assert doc.source_type == "html"
        assert doc.source == "https://example.com/page"
        assert doc.timestamp
        assert doc.title == "Test Page"
        assert "Main content paragraph." in doc.content
        for excluded in ("Navigation", "Footer", "alert"):
            assert excluded not in doc.content
        assert doc.metadata["domain"] == "example.com"
        assert doc.metadata["path"] == "/page"
        assert doc.metadata["description"] == "A test page"
        assert doc.metadata["keywords"] == "test,unit"
        assert doc.metadata["og_title"] == "OG Test"

    def test_title_fallback_to_h1(self):
        doc = HTMLParser().parse(
            "<html><body><h1>Heading</h1><p>Content</p></body></html>",
            "https://example.com",
        )
        assert doc.title == "Heading"

    def test_custom_exclude_elements(self):
        doc = HTMLParser(exclude_elements=["p"]).parse(SAMPLE_HTML, "https://example.com")
        assert "Main content paragraph." not in doc.content

    def test_fetch_success_and_failure(self):
        import requests
        parser = HTMLParser()

        with patch.object(parser.session, "get") as mock_get:
            mock_resp = MagicMock(text="<html><body>OK</body></html>")
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            assert parser.fetch("https://example.com") == "<html><body>OK</body></html>"

        with patch.object(parser.session, "get", side_effect=requests.RequestException("fail")):
            assert parser.fetch("https://bad-url.com") is None

    def test_ingest_fetches_and_parses(self):
        parser = HTMLParser()
        with patch.object(parser, "fetch", return_value=SAMPLE_HTML):
            doc = parser.ingest("https://example.com")
            assert doc is not None and doc.title == "Test Page"

    def test_ingest_returns_none_on_fetch_failure(self):
        parser = HTMLParser()
        with patch.object(parser, "fetch", return_value=None):
            assert parser.ingest("https://bad-url.com") is None


# ---------------------------------------------------------------------------
# Plugin registries (auto-discovery)
# ---------------------------------------------------------------------------

class TestRegistries:
    def test_parser_registry(self):
        from parsers import get_parser_class, supported_types
        assert "html" in supported_types()
        assert get_parser_class("html") is HTMLParser
        with pytest.raises(ValueError, match="Unknown parser"):
            get_parser_class("nonexistent")

    def test_embedder_registry(self):
        from embedders import supported_stores, get_embedder_class
        assert "qdrant" in supported_stores()
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder_class("nonexistent")


# ---------------------------------------------------------------------------
# QdrantEmbedder (mocked - no Qdrant or FastEmbed needed)
# ---------------------------------------------------------------------------

class TestQdrantEmbedder:
    @patch.dict(os.environ, {"VECTOR_DB_HOST": "myhost", "VECTOR_DB_PORT": "1234"})
    def test_init_from_env(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        assert embedder.host == "myhost"
        assert embedder.port == 1234
        assert embedder.store_type == "qdrant"

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_creates_collection_and_upserts(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        mock_model = MagicMock()

        # Use side_effect to return different embeddings for _ensure_collection and actual
        # embedding from mocked embedder
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2, 0.3])],  # _ensure_collection test embedding
            [np.array([0.1, 0.2, 0.3])],  # actual embedding
        ]
        MockTextEmbedding.return_value = mock_model

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        MockQdrantClient.return_value = mock_client

        assert QdrantEmbedder().store([_make_doc()], "new-col") == 1
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_skips_collection_creation_if_exists(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        MockTextEmbedding.return_value = MagicMock(
            embed=MagicMock(return_value=[np.array([0.1, 0.2, 0.3])])
        )
        existing = MagicMock()
        existing.name = "existing"
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[existing])
        MockQdrantClient.return_value = mock_client

        QdrantEmbedder().store([_make_doc()], "existing")
        mock_client.create_collection.assert_not_called()

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_batches_upserts(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        # Use side_effect to return different embeddings for _ensure_collection and actual
        # embeddings from mocked embedder
        mock_model = MagicMock()
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2, 0.3])],                          # _ensure_collection
            [np.array([0.1, 0.2, 0.3]) for _ in range(5)],        # actual
        ]
        MockTextEmbedding.return_value = mock_model

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        MockQdrantClient.return_value = mock_client

        docs = [_make_doc(source=f"https://example.com/{i}") for i in range(5)]
        assert QdrantEmbedder(batch_size=2).store(docs, "col") == 5
        assert mock_client.upsert.call_count == 3  # 2+2+1


# ---------------------------------------------------------------------------
# DataIngestor orchestrator
# ---------------------------------------------------------------------------

class TestDataIngestor:
    @patch("parsers.html_parser.HTMLParser.fetch")
    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_ingest_end_to_end(self, MockTextEmbedding, MockQdrantClient, mock_fetch):
        from ingestor import DataIngestor

        mock_fetch.return_value = SAMPLE_HTML
        MockTextEmbedding.return_value = MagicMock(
            embed=MagicMock(side_effect=[
                [np.array([0.1, 0.2, 0.3])],
                [np.array([0.1, 0.2, 0.3])],
            ])
        )
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        MockQdrantClient.return_value = mock_client

        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")
        count = ingestor.ingest(["https://example.com"], source_type="html",
                                store_type="qdrant", collection="test")

        assert count == 1
        mock_fetch.assert_called_once_with("https://example.com")
        mock_client.upsert.assert_called_once()

    def test_ingest_failed_parse_returns_zero(self):
        from ingestor import DataIngestor
        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")
        with patch.object(ingestor, "get_parser") as mock_gp:
            mock_gp.return_value = MagicMock(ingest=MagicMock(return_value=None))
            assert ingestor.ingest(["https://bad.com"]) == 0

    def test_config_loading(self, tmp_path):
        import yaml
        from ingestor import DataIngestor

        # Missing file -> empty config
        assert DataIngestor(config_path="/nonexistent/config.yaml").config == {}

        # Valid file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"request_delay": 2.0, "html": {"timeout": 10}}))
        ingestor = DataIngestor(config_path=str(config_file))
        assert ingestor.config["request_delay"] == 2.0
        assert ingestor.config["html"]["timeout"] == 10

    def test_detect_source_type(self):
        from ingestor import DataIngestor

        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")
        assert ingestor.detect_source_type('/path/to/file.pdf') == "pdf"
        assert ingestor.detect_source_type("https://example.com/page.html?q=1") == "html"
        assert ingestor.detect_source_type("https://example.com") == "html"
        # TLDs should not be treated as file extensions
        assert ingestor.detect_source_type("https://example.com/page") == "html"
        assert ingestor.detect_source_type("example.com") == "html"
        assert ingestor.detect_source_type("https://docs.example.org") == "html"
        assert ingestor.detect_source_type("https://api.example.io/v1") == "html"
