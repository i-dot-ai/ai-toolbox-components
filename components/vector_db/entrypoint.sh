#!/bin/bash
set -e

# 1. Start Qdrant in the background
# We use & to allow the script to continue to the next command
/qdrant/qdrant &
QDRANT_PID=$!

echo "Starting Qdrant (PID: $QDRANT_PID)..."

# 2. Wait for Qdrant to be ready
# It takes a few seconds for the binary to initialize and open the port
echo "Waiting for healthcheck on localhost:6333..."
MAX_RETRIES=30
COUNT=0

until curl -s http://localhost:6333/healthz > /dev/null; do
  COUNT=$((COUNT + 1))
  if [ $COUNT -ge $MAX_RETRIES ]; then
    echo "Error: Qdrant failed to start in time."
    exit 1
  fi
  sleep 1
done

echo "Qdrant is up! Running plugins..."

# Create directories if they don't exist
mkdir -p "/qdrant/config" "/qdrant/plugins"

# Copy default config files
if [ -d "/qdrant/defaults/config" ]; then
    echo "Processing default config files..."
    for file in /qdrant/defaults/config/*; do
        [ -e "$file" ] || continue
        base_file=$(basename "$file")
        dest_file="/qdrant/config/$base_file"
        
        if [ ! -f "$dest_file" ]; then
            echo "Copying default config: $base_file"
            cp "$file" "$dest_file"
        fi
    done
else
    echo "No default config directory found"
fi

# Copy default plugin files
if [ -d "/qdrant/defaults/plugins" ]; then
    echo "Processing default plugin files..."
    for file in /qdrant/defaults/plugins/*; do
        [ -e "$file" ] || continue
        base_file=$(basename "$file")
        dest_file="/qdrant/plugins/$base_file"
        
        if [ ! -f "$dest_file" ]; then
            echo "Copying default plugin: $base_file"
            cp "$file" "$dest_file"
        fi
    done
else
    echo "No default plugins directory found"
fi

# Run all Python plugins in the plugins directory
if ls /qdrant/plugins/*.py >/dev/null 2>&1; then
    echo "Running plugins..."
    for plugin in /qdrant/plugins/*.py; do
        [ -e "$plugin" ] || continue
        echo "Running plugin: $(basename "$plugin")"
        python3 "$plugin"
    done
else
    echo "No Python plugins found in plugins directory"
fi

echo "Setup complete. Keeping process alive..."

# 4. Bring the background process to the foreground
# This ensures that if Qdrant crashes, the container stops (as it should)
wait $QDRANT_PID
