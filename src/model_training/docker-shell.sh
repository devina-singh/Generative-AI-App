#!/bin/bash

set -e

# Build the image based on the Dockerfile
docker build -t data-generate-cli --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app \
training