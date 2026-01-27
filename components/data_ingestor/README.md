# Data Ingestor

A containerised service for ingesting content from various sources and embedding it into vector databases.

## Features

- Pluggable parser architecture - easily extend to support new content types
- Pluggable embedder architecture - support for multiple vector databases
- Auto-discovery of parser and embedder classes
- Simple CLI - just pass URLs as arguments
- Configurable via YAML and environment variables

## Supported Source Types

| Type | Description |
|------|-------------|
| `html` | Web pages fetched via URL |

## Supported Vector Stores

| Store | Description |
|-------|-------------|
| `qdrant` | Qdrant vector database |

## Usage

The data ingestor is designed to run alongside a vector database via docker compose. Use `docker compose run` to execute it within the compose network so it can reach the vector_db service by name.

```bash
# Ingest a single URL
docker compose run data_ingestor https://example.com

# Ingest multiple URLs
docker compose run data_ingestor \
  https://example.com \
  https://example.com/page2

# Ingest from a file
docker compose run \
  -v $(pwd)/urls.txt:/app/urls.txt \
  data_ingestor -f /app/urls.txt

# Specify collection name
docker compose run data_ingestor \
  -c my_collection \
  https://example.com
```

### Docker Compose

```yaml
services:
  vector_db:
    image: vector_db:latest
    ports:
      - "6333:6333"

  data_ingestor:
    image: data_ingestor:latest
    depends_on:
      - vector_db
    environment:
      - VECTOR_DB_HOST=vector_db
    volumes:
      - ./data/data_ingestor:/app/custom
```

## CLI Options

```
usage: ingestor.py [options] <source> [source ...]

positional arguments:
  sources               Sources to ingest (URLs, file paths, etc.)

options:
  -f, --file FILE       File containing sources (one per line)
  -t, --type TYPE       Source type (default: html)
  -s, --store STORE     Vector store type (default: qdrant)
  -c, --collection NAME Collection name (default: documents)
  --config PATH         Config file path
```

## Volume Mounts

| Path | Description |
|------|-------------|
| `/app/custom` | User customizations (defaults copied on first run) |

The custom directory contains:
- `config/` - Configuration files
- `parsers/` - Custom parser classes
- `embedders/` - Custom embedder classes

## Configuration

### Config File

Defaults are copied to `/app/custom/config/` on first run. Behavioral settings go here:

```yaml
# General settings
request_delay: 1.0

# Parser settings
html:
  user_agent: "MyBot/1.0"
  timeout: 30

# Embedder settings
qdrant:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32
```

### Environment Variables

Connection settings that vary between environments:

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_DB_HOST` | Qdrant server hostname | `localhost` |
| `VECTOR_DB_PORT` | Qdrant server port | `6333` |

## Adding New Parsers

Custom parsers can be added by placing Python files in the `/app/parsers` volume mount. On first run, the default parsers are copied there and can be used as examples.

1. Create a new file in the mounted `parsers/` directory (e.g., `pdf_parser.py`)

2. Implement a class inheriting from `BaseParser` that implments the core `source_type` property and the `parse` and `fetch` methods. E.g. for a PDF parser:


```python
from base import BaseParser, ParsedDocument

class PDFParser(BaseParser):
    @property
    def source_type(self) -> str:
        return "pdf"

    def parse(self, content: bytes, source: str) -> ParsedDocument:
        text = extract_pdf_text(content) # To be implemented

        return ParsedDocument(
            source=source,
            title="Extracted title",
            content=text,
            metadata={"pages": page_count},
            timestamp=self._current_timestamp(),
            source_type=self.source_type
        )

    def fetch(self, source: str) -> bytes:
        with open(source, "rb") as f:
            return f.read()
```

The parse function should return a ParsedDocument object containing the extracted text and metadata.

3. The parser is automatically discovered and registered on container restart

## Adding New Embedders

Custom embedders can be added by placing Python files in the `/app/embedders` volume mount.

1. Create a new file in the mounted `embedders/` directory (e.g., `pinecone_embedder.py`)

2. Implement a class inheriting from `BaseEmbedder` that implments the core `store_type` property and the `embed` and `store` methods. E.g. for a Pinecone embedder:

```python
from base import BaseEmbedder
from parsers.base import ParsedDocument

class PineconeEmbedder(BaseEmbedder):
    def __init__(self, api_key: str = "", environment: str = ""):
        self.api_key = api_key
        self.environment = environment

    @property
    def store_type(self) -> str:
        return "pinecone"

    def embed(self, text: str) -> list[float]:
        # Generate embedding vector suitable for store - code goes here
        return embedding_vector

    def store(self, documents: list[ParsedDocument], collection_name: str) -> int:
        # Store documents in Pinecone - code goes here
        return len(documents)
```

3. The embedder is automatically discovered and registered on container restart
