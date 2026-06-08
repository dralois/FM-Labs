#!/bin/bash
set -e

############################
# Enroot paths (deterministic)
############################
export ENROOT_CACHE_PATH="$HOME/enroot/cache"
export ENROOT_DATA_PATH="$HOME/enroot/data"

mkdir -p \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_DATA_PATH"

############################
# Configuration
############################

PORT=8888
IMAGE_SQSH="$ENROOT_CACHE_PATH/ubuntu-22.04.sqsh"
IMAGE_URI="docker://ubuntu:22.04"
CONTAINER_NAME="pytorch-jupyter"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

############################
# Import image (once)
############################
if [[ ! -f "$IMAGE_SQSH" ]]; then
  echo "Importing Docker image..."
  enroot import \
    --output "$IMAGE_SQSH" \
    "$IMAGE_URI"
else
  echo "Enroot image already exists."
fi

############################
# Create container (once)
############################
if [[ ! -d "$ENROOT_DATA_PATH/$CONTAINER_NAME" ]]; then
  echo "Creating container $CONTAINER_NAME..."
  enroot create \
    --name "$CONTAINER_NAME" \
    "$IMAGE_SQSH"
else
  echo "Container already exists."
fi

############################
# Start container + environment + Jupyter
############################
NODE_ID="$(hostname -s)"

echo
echo "================================================="
echo "LOCAL MACHINE:"
echo "ssh -L ${PORT}:${NODE_ID}:${PORT} <user-name>@uc3.scc.kit.edu"
echo "Open: http://localhost:${PORT}"
echo "================================================="
echo

enroot start \
  --root \
  --rw \
  --mount "$HOME:$HOME" \
  "$CONTAINER_NAME" \
  bash -lc "
    set -e
    cd \"$PROJECT_DIR\"

    # Minimal system deps for pixi installer and SSL
    apt-get update -qq && apt-get install -y -qq curl ca-certificates

    # Install pixi (official installer)
    if ! command -v pixi >/dev/null 2>&1; then
      curl -fsSL https://pixi.sh/install.sh | sh
    fi
    export PATH=\"\$HOME/.pixi/bin:\$PATH\"

    # Install GPU environment exactly from pixi.lock
    pixi install -e gpu

    # Start Jupyter in the GPU environment
    pixi run -e gpu jupyter lab \
      --no-browser \
      --port=$PORT \
      --ip=0.0.0.0 \
      --allow-root
  "
