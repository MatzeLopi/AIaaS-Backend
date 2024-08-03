# AIaaS - Backend

## Description
This repository contains the backend of the AIaaS project. The idea is to provide a simple API to deploy machine learning models. The API is built using FastAPI and the models are build and trained using TensorFlow.


## Installation

### Docker
1. Clone the repository
2. Be sure to have docker installed
3. Build image by running build_backend.sh
4. Run image by docker compose up

### Anaconda
1. Clone the repository
2. Be sure to have Anaconda installed
3. Create a new environment by running `conda env create -f environment.yml`
4. Activate the environment by running `conda activate AIaaS`
5. Run the server by running `fastapi dev main.py` 