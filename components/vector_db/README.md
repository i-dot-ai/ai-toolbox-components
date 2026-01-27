# Vector DB

A containerised Qdrant vector database with plugin support for automatic collection setup and configuration management.

## Features

- Based on official Qdrant image
- Automatic plugin execution on startup
- Default configuration and plugin copying to mounted volumes
- Health check endpoint at `/healthz`
- Customisable collection setup via Python plugins

## Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 6333 | HTTP | REST API |
| 6334 | gRPC | gRPC API |

## Usage

### Docker

```bash
# Build the image
docker build -t vector_db .

# Run with default configuration
docker run -p 6333:6333 -p 6334:6334 vector_db

# Run with persistent storage and custom config
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/data/vector_db:/app/custom \
  vector_db
```

### Docker Compose

```yaml
services:
  vector_db:
    image: vector_db:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/vector_db:/app/custom
```

## Volume Mounts

| Path | Description |
|------|-------------|
| `/app/custom` | User customizations (defaults copied on first run) |

The custom directory contains:
- `config/` - Configuration files
- `plugins/` - Python plugins executed on startup

## Plugins

Python scripts placed in `/app/custom/plugins/` are automatically executed after Qdrant starts. This enables custom collection setup, index configuration, and other initialisation tasks.

### Writing Custom Plugins

An `example_plugin.py` is included as a starting point.

Create a Python file in the plugins directory:

```python
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

def main():
    client = QdrantClient(
        host=os.getenv("VECTOR_DB_HOST", "localhost"),
        port=int(os.getenv("VECTOR_DB_PORT", 6333))
    )

    # Create custom collection
    client.create_collection(
        collection_name="my_collection",
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

if __name__ == "__main__":
    main()
```

## API Examples

```bash
# Health check
curl http://localhost:6333/healthz

# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/documents

# Create a collection
curl -X PUT http://localhost:6333/collections/my_collection \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 768, "distance": "Cosine"}}'

# Delete a collection
curl -X DELETE http://localhost:6333/collections/my_collection
```
