# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge Hub Blueprints - a repository of reusable, containerized AI components and template applications. Components are modular Docker services that can be independently developed and deployed. Applications are docker-compose orchestrations that combine components into complete solutions.

## Common Commands

### Install dependencies
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Build a component
```bash
docker compose build <component-name>
```

### Run a component
```bash
docker compose up -d <component-name>
```

### Run tests
```bash
# All tests
./run_tests.sh

# Component tests (builds and starts the component automatically)
./run_tests.sh component <component-name>

# Application tests
./run_tests.sh application <application-name>

# Run tests directly with pytest
uv run pytest -v tests/
uv run pytest -v tests/components/test_vector_db.py
```

### Run an application stack
```bash
docker compose --file applications/<app-name>/docker-compose.yaml up -d --build
```

## Architecture

### Directory Structure
- `components/` - Independent Docker services (e.g., `vector_db` wrapping Qdrant)
- `applications/` - Docker-compose files that orchestrate components into complete stacks
- `tests/` - Pytest-based tests split into `components/` and `applications/` subdirectories

### Component Structure
Each component in `components/<name>/` contains:
- `Dockerfile` - Build definition
- `entrypoint.sh` - Container startup script
- `src/` - Source code, configs, and plugins

### Test Fixtures
Tests use parametrized pytest fixtures defined in `tests/conftest.py`:
- `component_endpoint` - Builds, starts, and cleans up a single component; yields the service URL
- `application_endpoint` - Brings up an entire application stack and yields the main service URL

Test utilities in `tests/test_utils.py` provide:
- `wait_for_service()` - Wait for container to reach running state
- `verify_service_health()` - Wait for container health check to pass
- `get_application_services()` - Parse services from an application's docker-compose.yaml

### Test Naming Convention
- Component tests: `tests/components/test_<component_name>.py`
- Application tests: `tests/applications/test_<application_name>.py`

### Current Components
- `vector_db` - Qdrant vector database wrapper (ports 6333 HTTP, 6334 gRPC)

### Current Applications
- `mcp_datastore` - Vector database application using the vector_db component
