""" Internal API routes. """

# Built-in imports
import logging
from datetime import date

# Third-party imports
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

# Custom imports
from ..backend import crud, schemas
from ..backend.dependencies import get_db, verify_internal
from ..constants import DBError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal", tags=["internals"], dependencies=[Depends(verify_internal)]
)
