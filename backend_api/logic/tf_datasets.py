""" Infer Schema from CSV file. """

import logging
import polars as pl
from polars.datatypes import DataType
from fastapi import UploadFile

from ..constants import LOGLEVEL

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(__name__)


def infer_schema(
    file_path: str,
    has_headers: bool,
    sep: str = ",",
    decimal_comma: bool = False,
) -> dict[str, DataType]:
    """Infer the schema of a CSV file.

    Args:
        file_path (str): Path to uploaded file.
        has_headers (bool): Indicates if the CSV file has headers.
        sep (str, Optional): Separator used in the CSV file. Defaults to ",".
        decimal_comma (bool, Optional): Indicates if the CSV file uses a decimal comma. Defaults to False

    Returns:
        dict: Schema of the CSV file.
    """
    sample_df = pl.read_csv(
        file_path,
        has_header=has_headers,
        separator=sep,
        n_rows=100,
        n_threads=1,
        infer_schema=True,
        infer_schema_length=100,
        truncate_ragged_lines=True,
        decimal_comma=decimal_comma,
    )
    schema: pl.schema.Schema = sample_df.collect_schema()

    schema_dict: dict[str, pl.DataType] = {}

    for columns, dtype in schema.items():
        for column in columns.split(";"):
            column = column.strip()
            schema_dict[column] = dtype

    logger.debug(f"Schema: {schema_dict}")
    return schema_dict


def schema_to_sqlschema(schema: dict[str, DataType]) -> dict[str, str]:
    """Convert the schema to SQL schema.

    Args:
        schema (dict): Schema of the CSV file.

    Returns:
        dict: SQL schema of the CSV file. Maps column names to PostgreSQL data types.
    """
    sql_schema = {}
    for column, dtype in schema.items():
        if dtype == pl.Int64:
            sql_schema[column] = "INTEGER"
        elif dtype == pl.Float64:
            sql_schema[column] = "REAL"
        elif dtype == pl.Utf8:
            sql_schema[column] = "TEXT"
        elif dtype == pl.Date32:
            sql_schema[column] = "DATE"
        elif dtype == pl.Boolean:
            sql_schema[column] = "BOOLEAN"
        else:
            sql_schema[column] = "TEXT"

    logger.debug(f"SQL Schema: {sql_schema}")
    return sql_schema
