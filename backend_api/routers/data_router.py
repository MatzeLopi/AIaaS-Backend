"""  Routers for file and dataset upload and download. """

import logging
from uuid import uuid4

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..backend.crud import datasets
from ..backend import dependencies
from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..constants import LOGLEVEL
from ..logic.tf_datasets import save_upload_file

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger("Data Router")

router = APIRouter(
    prefix="/data",
    tags=["data"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)


@router.post("/upload")
async def upload_file(
    file: UploadFile,
    current_user: USER_DEPENDENCY,
    db: Session = Depends(get_db),
    has_headers: bool = True,
    delimiter: str = ",",
    decimal_comma: bool = True,
    dtype: str = "csv",
) -> str:
    """Upload a file to the server."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    else:
        file_path = await save_upload_file(file)
        fid = uuid4().hex
        await datasets.save_dataset_to_db(
            db,
            fid,
            user=current_user,
            dataset_path=file_path,
            dataset_type=dtype,
        )
        return fid


@router.get("/datasets")
async def get_datasets(
    current_user: USER_DEPENDENCY,
    db: Session = Depends(get_db),
):
    """Get the datasets for the current user."""
    return await datasets.get_available_datasets(db, current_user)


@router.get("/datasets/versions")
async def get_dataset_versions(
    current_user: USER_DEPENDENCY,
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Get the versions of a dataset."""
    return await datasets.get_dataset_versions(db, dataset_id, current_user)


@router.get("/dataset")
async def get_dataset(
    dataset_id: str,
    current_user: USER_DEPENDENCY,
    db: Session = Depends(get_db),
):
    """Get the latest version of a dataset."""
    return await datasets.get_dataset(db, dataset_id, current_user)


@router.get("/dataset/version")
async def get_dataset_version(
    dataset_id: str,
    version: int,
    current_user: USER_DEPENDENCY,
    db: Session = Depends(get_db),
):
    """Get a specific version of a dataset."""
    return await datasets.get_dataset_version(db, dataset_id, version, current_user)
