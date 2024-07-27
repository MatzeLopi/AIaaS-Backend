"""  Routers for file and dataset upload and download. """ 

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile

from ..backend import crud, schemas
from ..backend.dependencies import get_db, verify_password
from ..constants import DBError

from sqlalchemy.orm import Session

router = APIRouter(prefix="/data", tags=["data"], dependencies=[Depends(verify_password)])

@router.post("/upload")
async def upload_file(file: UploadFile, db: Session = Depends(get_db)) -> str:
    """ Upload a file to the server. """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    else:
        # DO Something with the file

        return file.filename