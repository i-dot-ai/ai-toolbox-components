# Use the official Qdrant image
FROM qdrant/qdrant:latest

# Install python to run the initialization script
RUN apt-get update && apt-get install -y --no-install-recommends \
python3 python3-pip python3-venv curl

# Install Qdrant client
RUN pip install qdrant-client --break-system-packages && \
    rm -rf /var/lib/apt/lists/*

# Expose default ports
# 6333 - HTTP API
# 6334 - gRPC API
EXPOSE 6333 6334

# Copy your configuration and init scripts
COPY entrypoint.sh /qdrant/entrypoint.sh
COPY src/ /qdrant/defaults/

WORKDIR /qdrant

# 5. Define the Healthcheck so the user's other services can "wait" for it
HEALTHCHECK --interval=5s --timeout=3s --retries=5 \
  CMD curl -f http://localhost:6333/healthz || exit 1

# Make the entrypoint executable
RUN chmod +x /qdrant/entrypoint.sh

ENTRYPOINT ["/qdrant/entrypoint.sh"]