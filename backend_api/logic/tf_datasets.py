""" Infer Schema from CSV file. """

import polars as pl
from polars.datatypes import DataType
from fastapi import UploadFile


def infer_schema(
    data: UploadFile, has_headers: bool, sep: str = ","
) -> dict[str, DataType]:
    """Infer the schema of a CSV file.

    Args:
        data (UploadFile): CSV file to infer the schema from.
        has_headers (bool): Indicates if the CSV file has headers.
        sep (str): Separator used in the CSV file. Defaults to ",".

    Returns:
        dict: Schema of the CSV file.
    """
    sample_df = pl.read_csv(
        data.file,
        has_header=has_headers,
        separator=sep,
        n_rows=100,
        n_threads=1,
        infer_schema= True,
        infer_schema_length=100,
        truncate_ragged_lines=True,
    )
    schema:pl.schema.Schema = sample_df.collect_schema()
    return schema


def schema_to_sqlschema(schema: dict[str, DataType]) -> dict[str, str]:
    """Convert the schema to SQL schema.

    Args:
        schema (dict): Schema of the CSV file.

    Returns:
        dict: SQL schema of the CSV file. Maps column names to PostgreSQL data types.
    """
    sql_schema = {}
    for columns, dtype in schema.items():
        for column in columns.split(";"):
            column = column.strip()
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

    return sql_schema
