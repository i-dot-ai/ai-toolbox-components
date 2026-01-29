"""
Integration tests for the data_ingestor component.

Requires Docker to build and run the data_ingestor container.
The data_ingestor is a batch CLI tool (no HTTP server or healthcheck),
so we verify that the container starts and exits successfully.
"""

import subprocess

import pytest


def test_container_runs_with_help():
    """Test that data_ingestor container starts and prints help."""
    result = subprocess.run(
        ["docker", "run", "--rm", "data_ingestor", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout.lower() or "usage" in result.stdout.lower()
