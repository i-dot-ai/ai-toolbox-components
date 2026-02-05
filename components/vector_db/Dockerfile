# Use the official Qdrant image
FROM qdrant/qdrant:latest

# Install python to run the initialisation script
RUN apt-get update && apt-get install -y --no-install-recommends \
python3 python3-pip python3-venv curl

# Install Qdrant client and PyYAML
RUN pip install qdrant-client pyyaml --break-system-packages && \
    rm -rf /var/lib/apt/lists/*

# Expose default ports
# 6333 - HTTP API
# 6334 - gRPC API
EXPOSE 6333 6334

WORKDIR /app

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy customizable defaults (will be copied to /app/custom on startup if not present)
COPY src/config/ /app/defaults/config/
COPY src/plugins/ /app/defaults/plugins/

# Create custom directory (single mount point for user customization)
RUN mkdir -p /app/custom

# Healthcheck so dependent services can wait for readiness
HEALTHCHECK --interval=5s --timeout=3s --retries=5 \
  CMD curl -f http://localhost:6333/healthz || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
