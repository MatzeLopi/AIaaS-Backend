import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated
from passlib.context import CryptContext


import jwt
from jwt import InvalidTokenError
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy.ext.asyncio import AsyncSession
from asyncpg import connect

from .database import sessionmanager
from . import models, schemas
from .crud.users import get_user
from ..constants import ALGORITHM


logger = logging.getLogger(__name__)


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "KEY"
INTERNAL_KEY = "SOMEKEY"


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_db():
    """Get a database session.

    Returns:
        Session: Database session.

    """
    async with sessionmanager.session() as session:
        yield session


class Token(BaseModel):
    """Token model.

    Attributes:
        access_token (str): JWT token.
        token_type (str): Type of token.
    """

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model.

    Attributes:
        username (str): Username of the user.
    """

    username: str | None = None


async def verify_internal(internal_key=Header()) -> bool:
    """Verify that internal requests are allowed to be made.

    Args:
        key (str): Key to validate.

    Raises:
        HTTPException: Raised if the key is invalid.

    Returns:
        bool: True if the key is valid.
    """
    if internal_key != INTERNAL_KEY:
        logger.debug(f"Invalid internal key: {internal_key}")
        raise HTTPException(status_code=401, detail="Unauthorized request")
    else:
        return True


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify the password against the hashed password.

    Args:
        plain_password (str): Password supplied by the user.
        hashed_password (str): Password stored in the DB.

    Returns:
        bool: Indicates if the password is correct.
    """
    logger.debug(
        f"Verifying password: {pwd_context.verify(plain_password, hashed_password)}"
    )
    return pwd_context.verify(plain_password, hashed_password)


async def authenticate_user(
    username: str, password: str, db: AsyncSession
) -> models.User:
    """Authenticate the user.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.
        db (Session): Database session.

    Raises:
        HTTPException: Raised if the credentials are invalid.

    Returns:
        User: User object if the credentials are valid.
    """

    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    logger.debug(f"Authenticating user: {username}")

    try:
        user: models.User = await get_user(db, username)
    except ValueError:
        logger.debug(f"User not found: {username}")
        raise auth_error
    else:
        if not await verify_password(password, user.hashed_password):
            logger.debug("Password verification failed")
            raise auth_error
    return user


async def create_access_token(
    data: dict, expires_delta: timedelta | None = None
) -> str:
    """Create an access token for the user.

    Args:
        data (dict): Data of the request which will be encoded in the token.
        expires_delta (timedelta | None, optional): _description_. Defaults to 20 Minutes.

    Returns:
        str: JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=20)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], db: AsyncSession = Depends(get_db)
):
    """Get the current user from the token.

    Args:
        token (Annotated[str, Depends): JWT token.

    Raises:
        credentials_exception: Raised if the credentials are invalid.

    Returns:
        _type_: _description_
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.debug("Username not found in token")
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        logger.debug(f"Invalid token: {token}")
        raise credentials_exception

    try:
        user = await get_user(db, username=token_data.username)
    except ValueError:
        logger.debug(f"User not found: {token_data.username}")
        raise credentials_exception
    else:
        return user


async def get_current_active_user(
    current_user: Annotated[schemas.UserBase, Depends(get_current_user)],
):
    """Check if the current user is active.

    Args:
        current_user (Annotated[User, Depends): Current user, supplied by get_current_user.

    Raises:
        HTTPException: Raised if the user is inactive.

    Returns:
        User: User object if the user is active.
    """

    if current_user.disabled:
        logger.debug(f"User {current_user.username} is inactive")
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


USER_DEPENDENCY = Annotated[schemas.UserBase, Depends(get_current_active_user)]
