# Use official Python slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    beautifulsoup4 \
    pyyaml \
    lxml \
    "qdrant-client[fastembed]"

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy application source
COPY src/ingestor.py src/registry.py /app/

# Copy customizable defaults (will be copied to /app/custom on startup if not present)
COPY src/config/ /app/defaults/config/
COPY src/parsers/ /app/defaults/parsers/
COPY src/embedders/ /app/defaults/embedders/

# Create custom directory (single mount point for user customization)
RUN mkdir -p /app/custom

# Set PYTHONPATH so custom parsers/embedders take precedence
ENV PYTHONPATH="/app/custom:/app"

ENTRYPOINT ["/app/entrypoint.sh"]
