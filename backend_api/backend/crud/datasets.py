""" CRUD file for Datasets. """

import logging
from uuid import uuid4

import polars as pl

from fastapi import BackgroundTasks, UploadFile, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .. import models
from ...constants import CHUNK_SIZE


def create_table_from_file(
    db: Session, schema:dict[str,str], user: models.User
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


    columns = ", ".join([f"{name} {dtype}" for name, dtype in schema.items()])
    table_name = f"table_{uuid4()}"

    sql = f"CREATE TABLE {table_name} ({columns})"
    metadata = models.UserTable(username=user.username, table_name=table_name)

    try:
        with db.begin():
            db.execute(sql)

            db.add(metadata)

    except SQLAlchemyError as e:
        logging.error(f"Error during transaction: {e}")
        raise HTTPException(status_code=400, detail="Could not create table")

    else:
        return table_name


def insert_csv(
    db: Session,
    table_name: str,
    data: UploadFile,
    csv_schema: dict,
    has_headers: bool,
    sep: str = ",",
) -> bool:
    """Insert data from a CSV file into the database.

    Args:
        db (Session): Session object
        table_name (str): Name of the table to insert data into.
        data (UploadFile): CSV file to insert data from.
        csv_schema (dict): Schema of the CSV file.
        has_headers (bool): Indicates if the CSV file has headers.
        sep (str): Separator used in the CSV file. Defaults to ",".

    Raises:
        HTTPException: Raised if the data could not be inserted.

    Returns:
        bool: True if the data was inserted successfully.
    """
    lazy_frame = pl.scan_csv(
        data.file,
        low_memory=True,
        has_header=has_headers,
        separator=sep,
        infer_schema=False,
        schema=csv_schema,
    )
    for chunk in lazy_frame.chunks(streaming=True, chunk_size=CHUNK_SIZE):
        try:
            chunk.write_database(table_name, db, if_exists="append")
        except Exception as e:
            logging.error(f"Error during insert: {e}")
            raise HTTPException(status_code=400, detail="Could not insert data")
        else:
            return True
