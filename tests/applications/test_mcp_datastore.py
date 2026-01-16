import pytest
import os

from tests.test_utils import wait_for_service
from tests.test_utils import verify_service_health
from tests.test_utils import get_application_services

app_name = "mcp_datastore"
port = 6333

@pytest.mark.parametrize("application_endpoint", [(app_name, port)], indirect=True)
def test_service_health(application_endpoint):
    services = get_application_services(app_name)
    for service in services:
        assert verify_service_health(service), "Unhealthy services detected"


@pytest.mark.parametrize("application_endpoint", [(app_name, port)], indirect=True)
def test_endpoints(application_endpoint):
    # Real endpoint tests would go here
    assert True
