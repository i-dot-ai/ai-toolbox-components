# MCP Datastore

A vector database application for ingesting and storing document embeddings. Combines the `vector_db` and `data_ingestor` components to provide a complete content ingestion and retrieval pipeline.

## Components

| Component | Description |
|-----------|-------------|
| [vector_db](/components/vector_db/README.md) | Qdrant vector database with plugin support |
| [data_ingestor](/components/data_ingestor/README.md) | Content ingestion and embedding service |

## Features

- Automatic collection setup via vector_db plugin
- HTML content parsing
- Vector embedding generation using sentence-transformers
- Configurable parsers and embedders via mounted volumes

## Prerequisites

Build the component images before running the application:

```bash
docker compose build vector_db
docker compose build data_ingestor
```

## Usage

Run all commands from the `applications/mcp_datastore` directory.

### Start the Vector Database

```bash
docker compose up -d vector_db
```

### Ingest Content

Use `docker compose run` to execute the data ingestor within the compose network:

```bash
# Ingest a single URL
docker compose run data_ingestor https://example.com

# Ingest multiple URLs
docker compose run data_ingestor \
  https://example.com \
  https://example.com/page2

# Ingest from a file
docker compose run -v $(pwd)/urls.txt:/app/urls.txt \
  data_ingestor -f /app/urls.txt

# Specify a collection name
docker compose run data_ingestor -c my_collection https://example.com
```

### Ingest Local Files

To ingest local files, place them in an `input/` directory within your working directory. The data_ingestor container mounts this directory automatically:

```bash
# Create the input directory and add files
mkdir -p input
cp my_document.html input/

# Ingest a local file
docker compose run data_ingestor input/my_document.html

# Ingest multiple local files
docker compose run data_ingestor input/doc1.html input/doc2.html
```

**Note:** Only files within the `input/` directory are accessible to the data_ingestor container.

### Query the Database

```bash
# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/documents

# Health check
curl http://localhost:6333/healthz
```

### Stop the Application

```bash
docker compose down
```

## Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 6333 | HTTP | Qdrant REST API |
| 6334 | gRPC | Qdrant gRPC API |

## Configuration

Configuration files are mounted from the `code/` directory:

| Path | Description |
|------|-------------|
| `code/vector_db/config/` | Vector database configuration |
| `code/vector_db/plugins/` | Startup plugins (e.g., collection setup) |
| `code/data_ingestor/config/` | Ingestor configuration |
| `code/data_ingestor/parsers/` | Custom parser classes |
| `code/data_ingestor/embedders/` | Custom embedder classes |

### Default Configuration

The data ingestor is configured with:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- **Collection**: `documents` (created automatically by the setup plugin)
- **Distance metric**: Cosine similarity

### Customization

To modify the embedding model or collection settings:

1. Edit `code/data_ingestor/config/config.yaml` for ingestor settings
2. Add e.g. `code/vector_db/plugins/setup_collections.py` for collection configuration
3. Restart the services

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  data_ingestor  │────▶│    vector_db    │
│                 │     │    (Qdrant)     │
│  - HTML parser  │     │                 │
│  - Embedder     │     │  - Collections  │
└─────────────────┘     │  - Plugins      │
                        └─────────────────┘
                              │
                        ┌─────┴─────┐
                        │  :6333    │
                        │  :6334    │
                        └───────────┘
```

## Resource Limits

| Service | Memory Limit | CPU Limit |
|---------|--------------|-----------|
| vector_db | 4GB | 2 cores |
| data_ingestor | 2GB | 1 core |
