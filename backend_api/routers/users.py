""" User router for the FastAPI app """

import logging
import traceback
from datetime import timedelta
from typing import Annotated

# Third-party modules
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from sqlalchemy.orm import Session

# Custom modules
from ..constants import ACCESS_TOKEN_EXPIRE_MINUTES
from ..backend import crud, schemas
from ..backend import dependencies
from ..backend.dependencies import Token, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/new/", response_model=schemas.UserBase)
def create_user_endpoint(
    background_task: BackgroundTasks,
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
):
    try:
        message = crud.create_user(db, user, background_task)
    except Exception as e:
        logger.error(
            f"Could not create user: {e}. Trackeback: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=400, detail="Could not create user")
    else:
        return message


@router.get("/verify_email/{verify_token}")
def verify_email(verify_token: str, db: Session = Depends(get_db)) -> Token:
    try:
        user = crud.verify_email(db, verify_token)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    else:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = dependencies.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return Token(access_token=access_token, token_type="bearer")


@router.get("/me/", response_model=schemas.UserBase)
async def read_users_me(
    current_user: Annotated[
        schemas.UserBase, Depends(dependencies.get_current_active_user)
    ],
):
    return current_user


@router.delete("/delete", response_model=schemas.UserBase)
def delete_user(
    password: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(dependencies.get_current_active_user),
):
    """Function to delete a user from the database

    Args:
        password (str): Password of the user.

    Returns:
        User: User object if the user is deleted.

    Raises:
        HTTPException: Raised if the password is incorrect and user is not deleted.
    """
    user = crud.get_user(db, current_user.username)

    if dependencies.verify_password(password, user.hashed_password):
        return crud.delete_user(db, current_user.username)
    else:
        raise HTTPException(status_code=400, detail="Incorrect password")
