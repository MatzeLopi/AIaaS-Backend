import logging
import os

ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Expire time for the access token in minutes.

LOGLEVEL = logging.DEBUG

ALGORITHM = "HS256"
UPLOAD_FILE_DIR = r"/temp/uploaded_files"  # Directory for uploaded files.
MODEL_DIR = r"/models"  # Directory for storing models.
CHUNK_SIZE = 1  # Chunk size for reading files in Rows.
TF_ENABLE_ONEDNN_OPTS = 0  # Enable OneDNN optimizations for TensorFlow.


class DBError(Exception):
    pass


if not os.path.exists(UPLOAD_FILE_DIR):
    os.makedirs(UPLOAD_FILE_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)