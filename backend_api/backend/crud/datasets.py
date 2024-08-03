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
from ...constants import UPLOAD_FILE_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile) -> Path:
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
