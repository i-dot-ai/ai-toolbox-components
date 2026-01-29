"""
Unit tests for the data_ingestor component.

Tests parsers, embedders, and the ingestor orchestrator using mocks
so no Docker or external services are needed.
"""

import hashlib
import sys
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

import pytest
import numpy as np

# Add data_ingestor src to path so we can import directly
_src = str(Path(__file__).resolve().parents[2] / "components" / "data_ingestor" / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Mock heavy dependencies that aren't installed locally (fastembed, qdrant_client)
# These must be mocked before importing embedders package
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

from parsers.base import BaseParser, ParsedDocument
from parsers.html_parser import HTMLParser
from embedders.base import BaseEmbedder


# ---------------------------------------------------------------------------
# ParsedDocument tests
# ---------------------------------------------------------------------------

class TestParsedDocument:
    def test_creation(self):
        doc = ParsedDocument(
            source="https://example.com",
            title="Example",
            content="Hello world",
            metadata={"domain": "example.com"},
            timestamp="2025-01-01T00:00:00Z",
            source_type="html",
        )
        assert doc.source == "https://example.com"
        assert doc.title == "Example"
        assert doc.content == "Hello world"
        assert doc.source_type == "html"

    def test_to_dict(self):
        doc = ParsedDocument(
            source="src", title="t", content="c",
            metadata={}, timestamp="ts", source_type="html",
        )
        d = doc.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "src"
        assert d["content"] == "c"


# ---------------------------------------------------------------------------
# HTMLParser tests
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


class TestHTMLParser:
    def test_source_type(self):
        parser = HTMLParser()
        assert parser.source_type == "html"

    def test_parse_extracts_title(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com/page")
        assert doc.title == "Test Page"

    def test_parse_removes_excluded_elements(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com/page")
        assert "Navigation" not in doc.content
        assert "Footer" not in doc.content
        assert "alert" not in doc.content

    def test_parse_extracts_main_content(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com/page")
        assert "Main content paragraph." in doc.content

    def test_parse_extracts_metadata(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com/page")
        assert doc.metadata["domain"] == "example.com"
        assert doc.metadata["path"] == "/page"
        assert doc.metadata["description"] == "A test page"
        assert doc.metadata["keywords"] == "test,unit"
        assert doc.metadata["og_title"] == "OG Test"

    def test_parse_sets_source_type(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com")
        assert doc.source_type == "html"

    def test_parse_sets_timestamp(self):
        parser = HTMLParser()
        doc = parser.parse(SAMPLE_HTML, "https://example.com")
        assert doc.timestamp  # non-empty

    def test_parse_title_fallback_to_h1(self):
        html = "<html><body><h1>Heading Title</h1><p>Content</p></body></html>"
        parser = HTMLParser()
        doc = parser.parse(html, "https://example.com")
        assert doc.title == "Heading Title"

    def test_parse_no_title(self):
        html = "<html><body><p>Just content</p></body></html>"
        parser = HTMLParser()
        doc = parser.parse(html, "https://example.com")
        assert doc.title == ""

    def test_custom_exclude_elements(self):
        parser = HTMLParser(exclude_elements=["p"])
        doc = parser.parse(SAMPLE_HTML, "https://example.com")
        assert "Main content paragraph." not in doc.content

    def test_fetch_success(self):
        parser = HTMLParser()
        with patch.object(parser.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "<html><body>OK</body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            result = parser.fetch("https://example.com")
            assert result == "<html><body>OK</body></html>"

    def test_fetch_failure_returns_none(self):
        import requests
        parser = HTMLParser()
        with patch.object(parser.session, "get", side_effect=requests.RequestException("fail")):
            result = parser.fetch("https://bad-url.com")
            assert result is None

    def test_ingest_with_content(self):
        parser = HTMLParser()
        doc = parser.ingest("https://example.com", content=SAMPLE_HTML)
        assert doc is not None
        assert doc.title == "Test Page"

    def test_ingest_fetch_fails_returns_none(self):
        parser = HTMLParser()
        with patch.object(parser, "fetch", return_value=None):
            doc = parser.ingest("https://bad-url.com")
            assert doc is None


# ---------------------------------------------------------------------------
# Parser registry tests
# ---------------------------------------------------------------------------

class TestParserRegistry:
    def test_html_parser_registered(self):
        from parsers import get_parser_class, supported_types
        assert "html" in supported_types()
        cls = get_parser_class("html")
        assert cls is HTMLParser

    def test_unknown_type_raises(self):
        from parsers import get_parser_class
        with pytest.raises(ValueError, match="Unsupported source type"):
            get_parser_class("nonexistent")


# ---------------------------------------------------------------------------
# QdrantEmbedder tests (mocked - no Qdrant or FastEmbed needed)
# ---------------------------------------------------------------------------

class TestQdrantEmbedder:
    def _make_doc(self, source="https://example.com", content="test content"):
        return ParsedDocument(
            source=source, title="Test", content=content,
            metadata={}, timestamp="2025-01-01T00:00:00Z", source_type="html",
        )

    @patch.dict(os.environ, {"VECTOR_DB_HOST": "myhost", "VECTOR_DB_PORT": "1234"})
    def test_reads_env_vars(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        assert embedder.host == "myhost"
        assert embedder.port == 1234

    def test_store_type(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        assert embedder.store_type == "qdrant"

    def test_generate_id_deterministic(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        id1 = embedder._generate_id("https://example.com")
        id2 = embedder._generate_id("https://example.com")
        assert id1 == id2
        assert id1 == hashlib.md5(b"https://example.com").hexdigest()

    def test_generate_id_differs_for_different_sources(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        id1 = embedder._generate_id("https://a.com")
        id2 = embedder._generate_id("https://b.com")
        assert id1 != id2

    def test_store_empty_list_returns_zero(self):
        from embedders.qdrant_embedder import QdrantEmbedder
        embedder = QdrantEmbedder()
        assert embedder.store([], "test-collection") == 0

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_documents(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([0.1, 0.2, 0.3])]
        MockTextEmbedding.return_value = mock_model

        # Mock Qdrant client
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        MockQdrantClient.return_value = mock_client

        # Also mock the test embedding for _ensure_collection
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2, 0.3])],  # test embedding in _ensure_collection
            [np.array([0.1, 0.2, 0.3])],  # actual embedding
        ]

        embedder = QdrantEmbedder()
        doc = self._make_doc()
        count = embedder.store([doc], "test-collection")

        assert count == 1
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_skips_collection_creation_if_exists(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([0.1, 0.2, 0.3])]
        MockTextEmbedding.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "existing"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        MockQdrantClient.return_value = mock_client

        embedder = QdrantEmbedder()
        doc = self._make_doc()
        embedder.store([doc], "existing")

        mock_client.create_collection.assert_not_called()

    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_store_batches_upserts(self, MockTextEmbedding, MockQdrantClient):
        from embedders.qdrant_embedder import QdrantEmbedder

        mock_model = MagicMock()
        embeddings = [np.array([0.1, 0.2, 0.3]) for _ in range(5)]
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2, 0.3])],  # test embedding
            embeddings,                     # actual embeddings
        ]
        MockTextEmbedding.return_value = mock_model

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        MockQdrantClient.return_value = mock_client

        embedder = QdrantEmbedder(batch_size=2)
        docs = [self._make_doc(source=f"https://example.com/{i}") for i in range(5)]
        count = embedder.store(docs, "test-collection")

        assert count == 5
        # 5 docs with batch_size=2 -> 3 upsert calls (2+2+1)
        assert mock_client.upsert.call_count == 3


# ---------------------------------------------------------------------------
# Embedder registry tests
# ---------------------------------------------------------------------------

class TestEmbedderRegistry:
    def test_qdrant_registered(self):
        from embedders import supported_stores, get_embedder_class
        assert "qdrant" in supported_stores()

    def test_unknown_store_raises(self):
        from embedders import get_embedder_class
        with pytest.raises(ValueError, match="Unknown store type"):
            get_embedder_class("nonexistent")


# ---------------------------------------------------------------------------
# DataIngestor orchestrator tests
# ---------------------------------------------------------------------------

class TestDataIngestor:
    @patch("parsers.html_parser.HTMLParser.fetch")
    @patch("embedders.qdrant_embedder.QdrantClient")
    @patch("embedders.qdrant_embedder.TextEmbedding")
    def test_ingest_end_to_end(self, MockTextEmbedding, MockQdrantClient, mock_fetch):
        from ingestor import DataIngestor

        # Mock fetch
        mock_fetch.return_value = SAMPLE_HTML

        # Mock embedding
        mock_model = MagicMock()
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2, 0.3])],
            [np.array([0.1, 0.2, 0.3])],
        ]
        MockTextEmbedding.return_value = mock_model

        # Mock Qdrant
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        MockQdrantClient.return_value = mock_client

        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")
        count = ingestor.ingest(
            ["https://example.com"],
            source_type="html",
            store_type="qdrant",
            collection="test",
        )

        assert count == 1
        mock_fetch.assert_called_once_with("https://example.com")
        mock_client.upsert.assert_called_once()

    def test_ingest_no_sources_returns_zero(self):
        from ingestor import DataIngestor
        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")

        with patch.object(ingestor, "get_parser") as mock_parser:
            mock_p = MagicMock()
            mock_p.ingest.return_value = None
            mock_parser.return_value = mock_p
            count = ingestor.ingest(["https://bad.com"])
            assert count == 0

    def test_supported_types_includes_html(self):
        from ingestor import DataIngestor
        assert "html" in DataIngestor.supported_types()

    def test_supported_stores_includes_qdrant(self):
        from ingestor import DataIngestor
        assert "qdrant" in DataIngestor.supported_stores()

    def test_config_loading_missing_file(self):
        from ingestor import DataIngestor
        ingestor = DataIngestor(config_path="/nonexistent/config.yaml")
        assert ingestor.config == {}

    def test_config_loading_valid_file(self, tmp_path):
        import yaml
        config = {"request_delay": 2.0, "html": {"timeout": 10}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        from ingestor import DataIngestor
        ingestor = DataIngestor(config_path=str(config_file))
        assert ingestor.config["request_delay"] == 2.0
        assert ingestor.config["html"]["timeout"] == 10
