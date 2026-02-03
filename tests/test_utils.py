import time
import docker
import sys
import yaml
import subprocess

client = docker.from_env()

def wait_for_service(container_name, timeout=60):
    """Wait until a Docker container is in running state"""
    start = time.time()
    while True:
        try:
            # Find the container by looking in full names for the service name
            container = next((c for c in client.containers.list() if container_name in c.name), None)
            if container.status == 'running':
                return True
        except docker.errors.NotFound:
            pass

        if time.time() - start > timeout:
            raise TimeoutError(f"Container {container_name} did not start within {timeout} seconds")

        time.sleep(1)


def container_is_running(container_name):
    """Check if a Docker container is running"""
    container = next((c for c in client.containers.list() if container_name in c.name), None)
    return container is not None


def verify_service_health(container_name, timeout=180):
    " Verify the health of the service given by container_name"

    start = time.time()
    while True:

        container = next((c for c in client.containers.list() if container_name in c.name), None)
        if container is None:
            raise ValueError(f"No running container found for service {container_name}")
        if 'Health' not in container.attrs['State']:
            raise ValueError(f"Container for service {container_name} does not have health status")

        health_status = container.attrs['State']['Health']['Status']
        if health_status == 'healthy':
            return True
        if time.time() - start > timeout:
            raise TimeoutError(f"Service {container_name} did not become healthy within {timeout} seconds")

        time.sleep(1)


def get_application_services(app_name):
    ''' Get the list of services defined in an application's docker-compose file'''
    compose_file = f"applications/{app_name}/docker-compose.yaml"
    try:
        with open(compose_file) as f:
            services = yaml.safe_load(f)['services'].keys()
    except FileNotFoundError:
        # Error if application docker compose file does not exist
        raise FileNotFoundError(f"Docker compose file for application {app_name} not found at {compose_file}")

    return services
