import pdb
import yaml
import pytest
import subprocess

from tests.test_utils import wait_for_service
from tests.test_utils import verify_service_health
from tests.test_utils import get_application_services


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
def application_endpoint(request):
    """Fixture to manage a complete application for tests"""
    app_name, service_port = request.param

    services = get_application_services(app_name)

    try:
        # Start the entire application stack
        subprocess.run(
            ["docker", "compose", "--file", f"applications/{app_name}/docker-compose.yaml", "up", "-d", "--build"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to start application {app_name}: {e}")

    # Wait for all services to be ready
    for service in services:
        wait_for_service(service)

    # Determine main service URL
    yield f"http://localhost:{service_port}"

    # Cleanup entire application
    subprocess.run(
        ["docker", "compose", "--file", f"applications/{app_name}/docker-compose.yaml", "down"],
        check=True
    )
