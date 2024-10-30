""" Router for user permissions and roles."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..backend import dependencies

router = APIRouter(
    prefix="/permissions",
    tags=["permissions"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)

