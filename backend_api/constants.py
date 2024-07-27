ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Expire time for the access token in minutes.

ALGORITHM = "HS256"

CHUNK_SIZE = 10000 # Chunk size for reading files in Rows.
class DBError(Exception):
    pass