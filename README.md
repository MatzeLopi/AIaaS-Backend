# AIaaS - Backend

## Description
This repository contains the backend of the AIaaS project. The idea is to provide a simple API to deploy machine learning models. The API is built using FastAPI and the models are build and trained using TensorFlow.


## Installation
1. Clone the repository

### Docker
   1. Be sure to have docker installed
   2. Run `build_run.sh` to build the image and run the container

### Anaconda
   1. Be sure to have Anaconda installed
   2. Create a new environment by running `conda env create -f environment.yml`
   3. Activate the environment by running `conda activate AIaaS`
   4. Be sure the DB is running
   5. Run the server by running `fastapi dev main.py` 