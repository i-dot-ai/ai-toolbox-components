import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_utils import wait_for_service


@pytest.fixture(scope="module")
def component_endpoint(request):
    """Fixture to manage a component service for tests"""
    service_name, internal_port = request.param

    # Build and start specific component
    subprocess.run(["docker", "compose", "build", service_name], check=True)
    subprocess.run(["docker", "compose", "up", "-d", service_name], check=True)

    # Wait for component to be ready
    wait_for_service(service_name)

    yield f"http://localhost:{internal_port}"

    # Cleanup component
    subprocess.run(["docker", "compose", "stop", service_name])
    subprocess.run(["docker", "compose", "rm", "-f", service_name])


@pytest.fixture(scope="module")
def application_endpoint(request, tmp_path_factory):
    """Set up a clean application directory, start all services, and yield the app dir."""
    app_name = request.param
    app_dir = tmp_path_factory.mktemp("app")

    # Copy docker-compose.yaml to clean directory
    src_compose = Path(f"applications/{app_name}/docker-compose.yaml")
    shutil.copy(src_compose, app_dir / "docker-compose.yaml")
    compose_file = app_dir / "docker-compose.yaml"

    # Start all services
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        cwd=app_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start application: {result.stderr}")

    yield app_dir

    # Cleanup containers
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down", "-v"],
        cwd=app_dir,
        capture_output=True,
    )


@pytest.fixture
def custom_parser_code():
    """Custom parser code for testing parser discovery."""
    return '''
"""Custom test parser for integration testing."""
import logging
from .base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class TestParser(BaseParser):
    """Simple parser for .test files."""

    @property
    def source_type(self) -> str:
        return "test"

    def fetch(self, source: str):
        logger.info(f"TestParser fetching: {source}")
        try:
            with open(source) as f:
                return f.read()
        except Exception as e:
            logger.error(f"TestParser failed to read {source}: {e}")
            return None

    def parse(self, content: str, source: str) -> ParsedDocument:
        logger.info(f"TestParser parsing: {source}")
        return ParsedDocument(
            source=source,
            title="Test Document",
            content=content,
            metadata={"parser": "test"},
            timestamp=self._current_timestamp(),
            source_type=self.source_type,
        )
'''
