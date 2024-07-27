"""  Routers for file and dataset upload and download. """

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..backend.crud import datasets
from ..backend import schemas, dependencies
from ..backend.dependencies import get_db, verify_password
from ..constants import DBError
from ..logic.tf_datasets import infer_schema, schema_to_sqlschema

router = APIRouter(
    prefix="/data", tags=["data"], dependencies=[Depends(verify_password)]
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
) -> str:
    """Upload a file to the server."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    else:

        schema = infer_schema(file.file)
        sql_schema = schema_to_sqlschema(schema)

        table_name = datasets.create_table_from_file(db, sql_schema, current_user)
        datasets.insert_csv(db, table_name, file, schema, has_headers, delimiter)

        return file.filename
