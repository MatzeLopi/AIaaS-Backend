""" Main entry point for the backend API."""

# Built-in modules
import logging
from typing import Annotated
from datetime import timedelta

# Third-party modules
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# Custom modules
from .constants import ACCESS_TOKEN_EXPIRE_MINUTES
from .backend import models
from .backend.database import engine
from .backend.dependencies import (
    get_db,
    authenticate_user,
    create_access_token,
    Token,
)

from .routers import users, internal

models.Base.metadata.create_all(bind=engine)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("backend_api")


app = FastAPI(title="AIaaS API", version="0.1.0")
app.include_router(users.router)
app.include_router(internal.router)


@app.post("/token")
def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
) -> Token:
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/")
async def root():
    return [{"message": "Hello World"}]
