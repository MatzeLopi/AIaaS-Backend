""" CRUD file for Datasets. """

import logging
from uuid import uuid4
from pathlib import Path
from shutil import copyfileobj
from asyncio import run

import polars as pl
import aiofiles
from fastapi import UploadFile, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from .. import models, schemas
from ...constants import CHUNK_SIZE, UPLOAD_FILE_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def insert_dataframe(db: Session, dataframe: pl.DataFrame, table_name: str) -> bool:
    """Insert a Polars DataFrame into the database.

    Args:
        db (Session): Session object
        dataframe (pl.DataFrame): DataFrame to insert.
        table_name (str): Name of the table to insert data into.

    Returns:
        bool: True if the data was inserted successfully.
    """
    # TODO: Fix this and run this with custom async code.
    # USE asyncpg to insert data


async def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Save an uploaded file to a temporary file.

    Args:
        upload_file (UploadFile): File to save.

    Returns:
        Path: Path to the saved file.
    """
    dir_path = Path(UPLOAD_FILE_DIR)
    suffix = Path(upload_file.filename).suffix
    file_name = str(uuid4().hex) + str(suffix)
    file_path = dir_path / file_name

    async with aiofiles.open(file_path, "wb+") as out_file:
        while content := await upload_file.read(2048):  # async read chunk
            print(content)
            await out_file.write(content)
    await upload_file.close()
    logger.debug(f"File saved to: {file_path}")

    return file_path


def delete_table(db: Session, table_name: str, user: schemas.UserBase) -> bool:
    """Delete a table from the database.

    Args:
        db (Session): Session object
        table_name (str): Name of the table to delete.

    Returns:
        bool: True if the table was deleted successfully.
    """

    if (
        not db.query(models.UserTable)
        .filter(
            models.UserTable.table_name == table_name,
            models.UserTable.username == user.username,
        )
        .first()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized request")
    try:
        db.execute(text(f"DROP TABLE {table_name}"))
        db.query(models.UserTable).filter(
            models.UserTable.table_name == table_name
        ).delete()
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Error during transaction: {e}")
        raise HTTPException(status_code=400, detail="Could not delete table")
    else:
        db.commit()
        logger.debug(f"Table {table_name} deleted")
        return True


def create_table_from_file(
    db: Session, schema: dict[str, str], user: schemas.UserBase
) -> str:
    """Create a table in the database.

    The created table depends on a list of columns and their types.

    This list is inferred from the schema of the uploaded file.

    Args:
        db (Session): Session object
        column_names (list[str]): List of column names.
        column_types (list[str]): List of columns and their types.

    Raises:
        HTTPException: Raised if the table could not be created.

    Returns:
        str: Name of the table created.
    """
    columns = ", ".join([f"{name.lower()} {dtype}" for name, dtype in schema.items()])
    table_name = f"table_{uuid4()}"
    table_name = table_name.replace("-", "_")

    sql = f"CREATE TABLE {table_name} ({columns})"
    metadata = models.UserTable(username=user.username, table_name=table_name)

    logger.debug(f"Creating table: {sql}")

    try:
        db.execute(text(sql))
        db.add(metadata)

    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Error during transaction: {e}")
        raise HTTPException(status_code=400, detail="Could not create table")

    else:
        db.commit()
        return table_name


def insert_csv(
    db: Session,
    table_name: str,
    file_path: str,
    csv_schema: dict,
    has_headers: bool,
    user: schemas.UserBase,
    sep: str = ",",
    decimal_comma: bool = False,
) -> bool:
    """Insert data from a CSV file into the database.

    Args:
        db (Session): Session object
        table_name (str): Name of the table to insert data into.
        file_path (str): Path to CSV file to upload.
        csv_schema (dict): Schema of the CSV file.
        has_headers (bool): Indicates if the CSV file has headers.
        user (schemas.UserBase): User object.
        sep (str, Optional): Separator used in the CSV file. Defaults to ",".
        decimal_comma (bool, Optional): Indicates if the CSV file uses a decimal comma. Defaults to False

    Raises:
        HTTPException: Raised if the data could not be inserted.

    Returns:
        bool: True if the data was inserted successfully.
    """
    # Check if user has access to the table
    if (
        not db.query(models.UserTable)
        .filter(
            models.UserTable.table_name == table_name,
            models.UserTable.username == user.username,
        )
        .first()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized request")

    # TODO: Port this to async
    lazy_frame = pl.scan_csv(
        file_path,
        has_header=has_headers,
        separator=sep,
        infer_schema=False,
        schema=csv_schema,
        decimal_comma=decimal_comma,
    )

    size: int = lazy_frame.select(pl.len()).collect().item()

    logger.debug(f"Size of the file: {size}")

    if size < CHUNK_SIZE:
        try:
            data_frame = lazy_frame.collect()
            insert_dataframe(db, data_frame, table_name)
        except Exception as e:
            logging.error(f"Error during insert: {e}")
            raise HTTPException(status_code=400, detail="Could not insert data")
        else:
            return True

    else:
        chunks = size // CHUNK_SIZE
        if size % CHUNK_SIZE != 0:
            chunks += 1

        # Read first chunk by defining new lazy frame
        lazy_frame = pl.scan_csv(
            file_path,
            has_header=has_headers,
            separator=sep,
            infer_schema=False,
            schema=csv_schema,
            decimal_comma=decimal_comma,
            n_rows=CHUNK_SIZE,
        )

        data_frame = lazy_frame.collect()
        try:
            insert_dataframe(db, data_frame, table_name)
        except Exception as e:
            logging.error(f"Error during insert: {e}")
            raise HTTPException(status_code=400, detail="Could not insert data")

        for chunk in range(1, chunks):
            lazy_frame = pl.scan_csv(
                file_path,
                skip_rows=CHUNK_SIZE * chunk,
                has_header=False,
                separator=sep,
                infer_schema=False,
                schema=csv_schema,
                decimal_comma=decimal_comma,
                n_rows=CHUNK_SIZE,
            )
            data_frame = lazy_frame.collect()
            try:
                insert_dataframe(db, data_frame, table_name)
            except Exception as e:
                logging.error(f"Error during insert: {e}")
                raise HTTPException(status_code=400, detail="Could not insert data")

    return True


def get_available_datasets(db: Session, user: schemas.UserBase):
    """Get all available datasets.

    Args:
        user (schemas.UserBase): User object.

    Returns:
        list[models.UserTable]: List of available datasets.
    """
    return (
        db.query(models.UserTable)
        .filter(models.UserTable.username == user.username)
        .all()
    )
