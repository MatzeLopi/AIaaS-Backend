#!/bin/bash

./build_backend.sh

docker compose -f docker-compose-backend.yml up -d
