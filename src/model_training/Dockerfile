# Use the official Debian-hosted Python image
FROM python:3.8-slim-buster

# Ensure we have an up to date baseline, install dependencies 
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app

WORKDIR /app

# Add Pipfile, Pipfile.lock
ADD Pipfile Pipfile.lock .env /app/ 

RUN pipenv sync

ENV PYENV_SHELL=/bin/bash

# Source code
ADD . /app

# Entry point
ENTRYPOINT ["/bin/bash"]

# Get into the pipenv shell
CMD ["-c", "pipenv shell"]