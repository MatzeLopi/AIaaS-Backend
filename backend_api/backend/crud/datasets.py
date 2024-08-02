""" CRUD file for Datasets. """

import logging
from uuid import uuid4
from pathlib import Path

import polars as pl
import aiofiles
from fastapi import UploadFile, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from .. import models, schemas
from ...constants import CHUNK_SIZE, UPLOAD_FILE_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def insert_dataframe(
    db: AsyncSession, dataframe: pl.DataFrame, table_name: str
) -> bool:
    """Insert a Polars DataFrame into the database.

    Args:
        db (AsyncSession): Session object
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


async def delete_table(
    db: AsyncSession, table_name: str, user: schemas.UserBase
) -> bool:
    """Delete a table from the database.

    Args:
        db (AsyncSession): Session object
        table_name (str): Name of the table to delete.

    Returns:
        bool: True if the table was deleted successfully.
    """

    if (
        not await db.query(models.UserTable)
        .filter(
            models.UserTable.table_name == table_name,
            models.UserTable.username == user.username,
        )
        .first()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized request")
    try:
        await db.execute(text(f"DROP TABLE {table_name}"))
        await db.query(models.UserTable).filter(
            models.UserTable.table_name == table_name
        ).delete()
    except SQLAlchemyError as e:
        await db.rollback()
        logging.error(f"Error during transaction: {e}")
        raise HTTPException(status_code=400, detail="Could not delete table")
    else:
        await db.commit()
        logger.debug(f"Table {table_name} deleted")
        return True


async def create_table_from_file(
    db: AsyncSession, schema: dict[str, str], user: schemas.UserBase
) -> str:
    """Create a table in the database.

    The created table depends on a list of columns and their types.

    This list is inferred from the schema of the uploaded file.

    Args:
        db (AsyncSession): Session object
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
        await db.execute(text(sql))
        db.add(metadata)

    except SQLAlchemyError as e:
        await db.rollback()
        logging.error(f"Error during transaction: {e}")
        raise HTTPException(status_code=400, detail="Could not create table")

    else:
        await db.commit()
        return table_name


async def insert_csv(
    db: AsyncSession,
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
        db (AsyncSession): Session object
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
    if not (
        await db.query(models.UserTable).filter(
            models.UserTable.table_name == table_name,
            models.UserTable.username == user.username,
        )
    ).first():
        raise HTTPException(status_code=401, detail="Unauthorized request")

    lazy_frame: pl.LazyFrame = pl.scan_csv(
        file_path,
        has_header=has_headers,
        separator=sep,
        infer_schema=False,
        schema=csv_schema,
        decimal_comma=decimal_comma,
    )

    size: int = await lazy_frame.select(pl.len()).collect_async().item()

    logger.debug(f"Size of the file: {size}")

    if size < CHUNK_SIZE:
        try:
            data_frame: pl.DataFrame = await lazy_frame.collect_async()
            await insert_dataframe(db, data_frame, table_name)
        except Exception as e:
            logging.error(f"Error during insert: {e}")
            raise HTTPException(status_code=400, detail="Could not insert data")
        else:
            return True

    else:
        chunks: int = size // CHUNK_SIZE
        if size % CHUNK_SIZE != 0:
            chunks += 1

        # Read first chunk by defining new lazy frame
        lazy_frame: pl.LazyFrame = pl.scan_csv(
            file_path,
            has_header=has_headers,
            separator=sep,
            infer_schema=False,
            schema=csv_schema,
            decimal_comma=decimal_comma,
            n_rows=CHUNK_SIZE,
        )

        data_frame = await lazy_frame.collect_async()
        try:
            await insert_dataframe(db, data_frame, table_name)
        except Exception as e:
            logging.error(f"Error during insert: {e}")
            raise HTTPException(status_code=400, detail="Could not insert data")

        for chunk in range(1, chunks):
            lazy_frame: pl.LazyFrame = pl.scan_csv(
                file_path,
                skip_rows=CHUNK_SIZE * chunk,
                has_header=False,
                separator=sep,
                infer_schema=False,
                schema=csv_schema,
                decimal_comma=decimal_comma,
                n_rows=CHUNK_SIZE,
            )
            data_frame: pl.DataFrame = await lazy_frame.collect_async()
            try:
                await insert_dataframe(db, data_frame, table_name)
            except Exception as e:
                logging.error(f"Error during insert: {e}")
                raise HTTPException(status_code=400, detail="Could not insert data")

    return True


async def get_available_datasets(db: AsyncSession, user: schemas.UserBase):
    """Get all available datasets.

    Args:
        db (AsyncSession): Session object
        user (schemas.UserBase): User object.

    Returns:
        list[models.UserTable]: List of available datasets.
    """
    return (
        await db.query(models.UserTable).filter(
            models.UserTable.username == user.username
        )
    ).all()
