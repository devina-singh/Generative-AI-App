#!/bin/bash

set -e

export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_NAME="ac215-rsvp-radar"
export GCP_PROJECT="ac215-project"
export GCP_ZONE="us-east1"

# Create the network if we don't have it yet
docker network inspect data-generate >/dev/null 2>&1 || docker network create data-generate

# Build the image based on the Dockerfile
docker build -t data-generate-cli --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name data-generate-cli -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v ~/.gitconfig:/etc/gitconfig \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
--network data-generate data-generate-cli