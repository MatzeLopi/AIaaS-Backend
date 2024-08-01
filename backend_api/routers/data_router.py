"""  Routers for file and dataset upload and download. """

import logging
from asyncio import run

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..backend.crud import datasets
from ..backend import schemas, dependencies
from ..backend.dependencies import get_db
from ..constants import DBError, LOGLEVEL
from ..logic.tf_datasets import infer_schema, schema_to_sqlschema

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger("Data Router")

router = APIRouter(
    prefix="/data",
    tags=["data"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)


@router.post("/upload")
def upload_file(
    file: UploadFile,
    current_user: Annotated[
        schemas.UserBase, Depends(dependencies.get_current_active_user)
    ],
    db: Session = Depends(get_db),
    has_headers: bool = True,
    delimiter: str = ",",
    decimal_comma: bool = True,
) -> str:
    """Upload a file to the server."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    else:
        file_path = run(datasets.save_upload_file_tmp(file))

    schema = infer_schema(file_path, has_headers, delimiter, decimal_comma)
    sql_schema = schema_to_sqlschema(schema)

    table_name = datasets.create_table_from_file(db, sql_schema, current_user)

    try:
        datasets.insert_csv(
            db,
            table_name,
            file_path,
            schema,
            has_headers,
            current_user,
            delimiter,
            decimal_comma,
        )
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        datasets.delete_table(db, table_name, current_user)
        raise HTTPException(
            status_code=400, detail={"error": "Error on upload", "message": str(e)}
        )
    else:
        return file.filename


@router.get("/datasets")
def get_datasets(
    current_user: Annotated[
        schemas.UserBase, Depends(dependencies.get_current_active_user)
    ],
    db: Session = Depends(get_db),
):
    """Get the datasets for the current user."""
    return datasets.get_available_datasets(db, current_user)
