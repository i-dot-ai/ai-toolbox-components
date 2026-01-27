import subprocess

import pytest

from tests.test_utils import verify_service_health
from tests.test_utils import get_application_services


APP_NAME = "mcp_datastore"


@pytest.mark.parametrize("application_endpoint", [APP_NAME], indirect=True)
class TestMcpDatastore:
    """Integration tests for the mcp_datastore application."""

    def test_service_health(self, application_endpoint):
        """Test that long-running services start and become healthy."""
        # data_ingestor is a run-once service, only check vector_db
        assert verify_service_health("vector_db"), "vector_db is unhealthy"

    def test_custom_parser_is_discovered_and_used(self, application_endpoint, custom_parser_code):
        """Test that a custom parser added to code/ is discovered and used."""
        compose_file = application_endpoint / "docker-compose.yaml"

        # Add a custom parser for .test files
        parser_dir = application_endpoint / "code" / "data_ingestor" / "parsers"
        assert parser_dir.exists(), "Entrypoint should have created parsers directory"
        (parser_dir / "test_parser.py").write_text(custom_parser_code)

        # Create a test file to ingest
        test_file = "/app/custom/sample.test"
        (parser_dir.parent / "sample.test").write_text("This is test content for the custom parser.")

        # Run data_ingestor with the test file
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "run", "--rm",
             "data_ingestor", test_file],
            cwd=application_endpoint,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr

        # Check that the custom parser was used
        assert "Creating parser: TestParser" in output, \
            f"Custom parser not loaded. Output:\n{output}"

        # Check that auto-detection selected the correct parser type
        assert "Parsing (test):" in output, \
            f"Parser auto-detection failed. Output:\n{output}"

        # Check that ingestion succeeded
        assert "Ingested 1 document(s)" in output, \
            f"Ingestion failed. Output:\n{output}"
