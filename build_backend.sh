#!/bin/bash
IMAGE_NAME="backend"
IMAGE_TAG="latest"

image=$IMAGE_NAME:$IMAGE_TAG

docker system prune -f

# Build the Docker image using the Dockerfile.backend
if docker build -t $image -f dockerfile .; then
    echo "Docker image $image built successfully"
else
    echo "Docker image $image failed to build"
    sleep 10
    exit 1
fi

# Save the Docker image to a tar file
if docker save -o $IMAGE_NAME.tar $image; then
    echo "Docker image $image saved to $IMAGE_NAME.tar"
else
    echo "Docker image $image failed to save to $IMAGE_NAME.tar"
    exit 1
fi
