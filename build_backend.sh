#!/bin/bash
IMAGE_NAME="backend"
IMAGE_TAG="latest"

# Build the Docker image using the Dockerfile.backend
docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f dockerfile.backend .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image build successful!"

    # Save the Docker image to the current folder
    docker save -o "./images/$IMAGE_NAME-$IMAGE_TAG.tar" "$IMAGE_NAME:$IMAGE_TAG"
    if [ $? -eq 0 ]; then
        echo "Docker image saved successfully!"
    else
        echo "Failed to save Docker image."
    fi

else
    echo "Docker image build failed."
fi

