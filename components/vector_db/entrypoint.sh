#!/bin/bash
set -e

# Custom directory (single mount point for user customizations)
CUSTOM_DIR="/app/custom"

# Subdirectories for customizable code
SUBDIRS=("config" "plugins")

# Copy defaults to custom directory if not already present
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "$CUSTOM_DIR/$dir"

    if [ -d "/app/defaults/$dir" ]; then
        for file in /app/defaults/$dir/*.py /app/defaults/$dir/*.yaml; do
            [ -e "$file" ] || continue
            base_file=$(basename "$file")
            dest_file="$CUSTOM_DIR/$dir/$base_file"

            if [ ! -f "$dest_file" ]; then
                echo "Copying default: $dir/$base_file"
                cp "$file" "$dest_file"
            fi
        done
    fi
done

# Load config options from config.yaml and export as Qdrant environment variables
# Qdrant uses the QDRANT__<SECTION>__<KEY> format (double underscores)
CONFIG_FILE="$CUSTOM_DIR/config/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading config from $CONFIG_FILE..."
    eval "$(python3 -c "
import yaml
def to_env_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f) or {}
for key, value in config.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            env_name = f'QDRANT__{key}__{sub_key}'.upper()
            print(f'export {env_name}=\"{to_env_value(sub_value)}\"')
    else:
        env_name = f'QDRANT__{key}'.upper()
        print(f'export {env_name}=\"{to_env_value(value)}\"')
")"
    echo "Config loaded."
else
    echo "No config file found at $CONFIG_FILE, using defaults."
fi

# Start Qdrant in the background (after config is loaded so env vars take effect)
/qdrant/qdrant &
QDRANT_PID=$!

echo "Starting Qdrant (PID: $QDRANT_PID)..."

# Wait for Qdrant to be ready
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

echo "Qdrant is up!"

# Run all Python plugins in the custom plugins directory
if ls "$CUSTOM_DIR/plugins/"*.py >/dev/null 2>&1; then
    echo "Running plugins..."
    for plugin in "$CUSTOM_DIR/plugins/"*.py; do
        [ -e "$plugin" ] || continue
        echo "Running plugin: $(basename "$plugin")"
        python3 "$plugin"
    done
else
    echo "No Python plugins found"
fi

echo "Setup complete. Keeping process alive..."

# Bring the background process to the foreground
wait $QDRANT_PID
