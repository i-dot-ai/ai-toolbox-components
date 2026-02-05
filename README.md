# AI Toolkit Components Repository

Build AI-powered applications faster with ready-to-use, containerised building blocks.

## What Is This?

This repository provides **components** and **applications** that you can use to quickly assemble AI solutions without starting from scratch.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            APPLICATIONS                                 │
│         Complete solutions built by combining components                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  mcp_datastore                                                    │  │
│  │  A document ingestion and semantic search application             │  │
│  │                                                                   │  │
│  │    ┌─────────────────┐          ┌─────────────────┐               │  │
│  │    │  data_ingestor  │ ──────▶  │    vector_db    │               │  │
│  │    │                 │          │                 │               │  │
│  │    │  • Parse HTML   │  embed   │  • Store vectors│               │  │
│  │    │  • Embed content│  ─────▶  │  • Search       │               │  │
│  │    │                 │          │  • Query API    │               │  │
│  │    └─────────────────┘          └─────────────────┘               │  │
│  │         COMPONENT                    COMPONENT                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

COMPONENTS = Independent, reusable Docker services (the building blocks)
APPLICATIONS = docker-compose orchestrations that wire components together
```

**Components** are standalone Docker services that do one thing well — like storing vectors or parsing documents. They're designed to be mixed and matched.

**Applications** are complete solutions that combine components using docker-compose. Copy an application's `docker-compose.yaml` to your project and you're ready to go.

## Getting Started

Choose your path:

| I want to... | Go to... |
|--------------|----------|
| **Use/adapt an existing application** | [User Guide → Applications](#applications) |
| **Use/adapt individual components** | [User Guide → Components](#components) |
| **Build new components or applications** | [Development Guide](#development-guide) |
| **Contribute back to the community** | [Contributing](#contributing) |

## User Guide

In order to employ the components and template applications provided in this repository, it is not necessary to clone the entire repository. Instead, you have two options:
1. **Template Application**: You can directly use the template applications available in the `applications/` directory. These applications are designed to showcase how to integrate and utilise the components effectively. By simply copying the relevant `docker-compose.yaml` files from the desired application directory, you can set up and run the application in your own environment without needing the full repository.
2. **Individual Components**: If you are interested in using specific components, you can write your own `docker-compose.yaml` file that references the components you wish to use and pulls the docker images as required. This allows you to tailor the setup to your specific needs without the overhead of the entire repository.

### Applications

The following applications are available:

| Application | Description |
|-------------|-------------|
| [mcp_datastore](applications/mcp_datastore/) | Document ingestion and semantic search |

To get started, navigate to the application's directory or copy its `docker-compose.yaml` to your project. Each application defines the services and components it uses.

Ensure you have Docker, Docker Desktop (or equivalent) and Docker Compose installed on your system. You can then run the application by executing the following command in the terminal from the application's directory:

```bash
docker compose up -d
```

This will initiate the process of pulling the necessary images and starting the services defined in the `docker-compose.yaml` file. When the containers run, they will write customisable code into the mounted volumes, allowing you to modify and extend the functionality as needed. For example:

```bash
$ cp applications/mcp_datastore/docker-compose.yaml .
$ docker compose up -d
```

There are some components (e.g. the [data_ingestor](components/data_ingestor/)) that are not continuously running services, but instead execute a task and then exit. For these, you can use `docker compose run` to execute them on demand. For example:

```bash
docker compose run data_ingestor http://example.com
```

Components are customisable via mounted volumes. Each component mounts a directory under `code` where defaults are copied on first run. Users can modify these files to customize behaviour:
- **vector_db**: `config/` for Qdrant settings, `plugins/` for startup scripts
- **data_ingestor**: `config/` for settings, `parsers/` and `embedders/` for custom code

After running the application for the first time, you can explore the `custom` directories to see the default configurations and code. Modify these files to tailor the components to your specific use case. For example, you might want to add custom parsers or embedders to the `data_ingestor`, or adjust the Qdrant settings in the `vector_db` configuration. For example, to edit the `data_ingestor` configuration:

```bash
cd code/data_ingestor/config/
vim config.yaml
```

### Components

The following components are available:

| Component | Description |
|-----------|-------------|
| [vector_db](components/vector_db/) | Qdrant vector database with plugin support |
| [data_ingestor](components/data_ingestor/) | Content ingestion and embedding |

Individual components can be used by creating a custom `docker-compose.yaml` file that references the desired component images. Below is an example of how to define a service using a component:

```yaml
version: '3.8'
services:
    my_component_service:
        image: ghcr.io/knowledge-hub/components/<component-a>:latest
        container_name: my_component_service
        ports:
            - "8080:8080"
        volumes:
            - ./data/my_component:/app/data
        environment:
            - CONFIG_PATH=/app/data/config.yaml
```

## Development Guide

If you wish to add or develop entirely new components or applications, or build off the source code directly, you can clone the entire repository. The structure is as follows:

```
.
├── applications/               # Application implementations using components
│   ├── <application-a>/        # Example application 1
│   │   ├── docker-compose.yaml # Application-specific service definitions
│   ├── <application-b>/        # Example application 2
├── components/                 # Independent service components
│   └── <component-a>/          # Example component service
│       ├── src/                # Application source code
│       ├── Dockerfile          # Component build definition
│       └── entrypoint.sh       # Container startup script
├── tests/                      # Test infrastructure
│   ├── applications/           # Application integration tests
│   │   └── test_<application-a>.py
│   ├── components/             # Component container tests (require Docker)
│   │   └── test_<component-a>.py
│   ├── unit/                   # Unit tests (no Docker needed)
│   │   └── test_<component-a>.py
│   ├── test_utils.py           # Shared testing utilities
│   └── conftest.py             # Pytest fixtures
├── .github/workflows/          # CI/CD pipeline definitions
├── docker-compose.yaml         # Local dev environment setup
└── LICENSE
```

The `components/` directory contains modular services that can be independently developed, tested, and deployed. The `applications/` directory showcases how these components can be orchestrated to build complete AI solutions.

### Quick Start

1. Ensure Docker and Docker Compose are installed on your system, along with Docker Desktop (or equivalent).

2. **Build and start services**:
   ```bash
   docker compose build <component-a>
   ```

3. **Launch the component**:
   ```bash
   docker compose up -d <component-a>
   ```

4. **Run tests**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   # Individual component tests
   ./run_tests.sh component <component-a>
   # Individual application tests
   ./run_tests.sh application <application-a>
   ```

5. **Customise/add new components**:
   - Create new services by adding `components/<component-new>/Dockerfile` and `entrypoint.sh`
   - Add/modify source code in `components/<component-a>/src/`
   - Add tests for new functionality under `tests/components/`

6. **Customise/add new application templates**:
   - Create new directories under `applications/`
   - Define services in `docker-compose.yaml` files
   - Add application-specific tests under `tests/applications/`


### Testing


#### Run tests locally

```bash
# Run unit tests (no Docker needed)
./run_tests.sh unit

# Run component tests (starts service automatically)
./run_tests.sh component vector_db

# Run application tests
./run_tests.sh application mcp_datastore
```

#### Adding new tests

Tests are written using `pytest`. To add new tests:
1. Add unit tests under `tests/unit/` named `test_<component_name>.py`
2. Add component tests under `tests/components/` named `test_<component_name>.py`
3. Add application tests under `tests/applications/` named `test_<application_name>.py`

### CI/CD Pipelines


The CI system runs three pipelines, each publishing test results directly on PRs:

1. **Unit Tests** (`.github/workflows/unit-tests.yml`):
   - Triggered on push to main and PRs
   - Runs all unit tests (no Docker needed)

2. **Component Build & Test** (`.github/workflows/component-build-test.yml`):
   - Triggered on push to main and PRs
   - Builds Docker images for all components
   - Runs component container tests

3. **Application Test** (`.github/workflows/application-test.yml`):
   - Runs after successful component builds
   - Tests full application stacks using built components

## Contributing

1. Create feature branches from `main`
2. Include tests for new functionality
3. Verify all GitHub Actions pass
4. Maintain documentation updates

See [LICENSE](LICENSE) for terms.
