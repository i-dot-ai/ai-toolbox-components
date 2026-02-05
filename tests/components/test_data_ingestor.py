"""
Integration tests for the data_ingestor component.

Requires Docker to build and run containers. Spins up vector_db via the
component_endpoint fixture so data_ingestor can store embeddings in a real
Qdrant instance.
"""

import os
import subprocess
import tempfile

import pytest
import requests

from tests.test_utils import verify_service_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_DB_PORT = "6333"
INGESTOR_SERVICE = "data_ingestor"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component_endpoint", [("vector_db", VECTOR_DB_PORT)], indirect=True)
class TestDataIngestorContainer:
    """Functional tests that exercise the full data_ingestor pipeline
    against a real Qdrant (vector_db) container."""

    def run_ingestor(self, *args, env_extra=None, timeout=120, volumes=None):
        """Run the data_ingestor container with the given CLI args.

        Uses the compose network so data_ingestor can reach vector_db
        by service name.
        """
        cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "VECTOR_DB_HOST=vector_db",
            "-e", f"VECTOR_DB_PORT={VECTOR_DB_PORT}",
        ]
        for k, v in (env_extra or {}).items():
            cmd += ["-e", f"{k}={v}"]
        for v in (volumes or []):
            cmd += ["-v", v]
        cmd += [INGESTOR_SERVICE, *args]

        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def test_help(self, component_endpoint):
        """Container starts and prints help text."""
        result = self.run_ingestor("--help")
        assert result.returncode == 0
        assert "ingest" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_ingest_single_url(self, component_endpoint):
        """Ingest a single URL and verify the document lands in Qdrant."""
        collection = "test-ingest-single"
        result = self.run_ingestor(
            "-c", collection,
            "https://example.com",
        )
        assert result.returncode == 0
        assert "stored 1" in result.stderr.lower() or "stored 1" in result.stdout.lower()

        # Verify the collection exists and has a point
        resp = requests.get(
            f"{component_endpoint}/collections/{collection}"
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["points_count"] == 1

    def test_ingest_multiple_urls(self, component_endpoint):
        """Ingest multiple URLs into the same collection."""
        collection = "test-ingest-multi"
        urls = [
            "https://example.com",
            "https://www.iana.org/help/example-domains",
        ]
        result = self.run_ingestor("-c", collection, *urls)
        assert result.returncode == 0

        resp = requests.get(
            f"{component_endpoint}/collections/{collection}"
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["points_count"] == len(urls)

    def test_ingest_from_file(self, component_endpoint):
        """Ingest URLs listed in a file."""
        collection = "test-ingest-file"
        # Create a temporary file with URLs and mount it into the container
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("https://example.com\n")
            f.write("https://www.iana.org/help/example-domains\n")
            tmp_path = f.name

        try:
            container_path = "/tmp/test_urls.txt"
            result = self.run_ingestor(
                "-c", collection,
                "--file", container_path,
                volumes=[f"{tmp_path}:{container_path}:ro"],
            )
            assert result.returncode == 0

            resp = requests.get(
                f"{component_endpoint}/collections/{collection}"
            )
            assert resp.status_code == 200
            assert resp.json()["result"]["points_count"] == 2
        finally:
            os.unlink(tmp_path)

    def test_ingest_idempotent(self, component_endpoint):
        """Ingesting the same URL twice should upsert, not duplicate."""
        collection = "test-ingest-idempotent"
        url = "https://example.com"

        self.run_ingestor("-c", collection, url)
        self.run_ingestor("-c", collection, url)

        resp = requests.get(
            f"{component_endpoint}/collections/{collection}"
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["points_count"] == 1

    def test_ingest_creates_searchable_embeddings(self, component_endpoint):
        """Verify stored embeddings are searchable via Qdrant scroll API."""
        collection = "test-ingest-search"
        self.run_ingestor("-c", collection, "https://example.com")

        # Scroll to retrieve stored points with payloads
        resp = requests.post(
            f"{component_endpoint}/collections/{collection}/points/scroll",
            json={"limit": 10, "with_payload": True, "with_vector": False},
        )
        assert resp.status_code == 200
        points = resp.json()["result"]["points"]
        assert len(points) == 1

        payload = points[0]["payload"]
        assert payload["source"] == "https://example.com"
        assert payload["source_type"] == "html"
        assert payload["title"]  # should have extracted a title
        assert payload["content"]  # should have extracted content

    def test_ingest_bad_url_exits_cleanly(self, component_endpoint):
        """Ingestor should handle unreachable URLs gracefully."""
        collection = "test-ingest-bad"
        result = self.run_ingestor(
            "-c", collection,
            "https://this-domain-does-not-exist-xyz.com",
        )
        # Should exit 0 (logs warning, stores 0 docs) rather than crash
        assert result.returncode == 0
        assert "no documents" in result.stderr.lower() or "stored 0" in result.stderr.lower()

    def test_ingest_unsupported_source_type_fails(self, component_endpoint):
        """Ingestor should fail with unknown parser for unsupported source types."""
        result = self.run_ingestor(
            "-c", "test-unsupported",
            "https://example.com/file.unsupported",
        )
        assert result.returncode != 0
        assert "unknown parser" in result.stderr.lower()
