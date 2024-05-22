#!/bin/bash
set -e

IMAGE_NAME="$1"
TAG="devel"

CUDA_IMAGE_VERSION="12.1.1-cudnn8"
UBUNTU_VERSION="22.04"

docker build \
  --build-arg TAG="$TAG" \
  --build-arg CUDA_IMAGE_VERSION="$CUDA_IMAGE_VERSION" \
  --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
  --no-cache --tag "${IMAGE_NAME}:${TAG}" --progress plain .

echo "Done"
