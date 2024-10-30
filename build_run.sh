#!/bin/bash

./build_backend.sh

docker compose -f docker-compose.yml up -d