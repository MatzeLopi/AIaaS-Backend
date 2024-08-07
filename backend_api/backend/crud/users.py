"""Create, Update, Delete, Read operations for users in the database."""

import logging

from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr

from .. import models, schemas
from ..utils import get_password_hash, generate_verification_token, send_email, retry


logger = logging.getLogger(__name__)


async def get_user(db: AsyncSession, username: str) -> models.User:
    """Get a user by name from the DB

    Args:
        db (AsyncSession): Session object
        username (str): Name of the user to get

    Raises:
        ValueError: If the user is not found

    Returns:
        models.User: User from the database
    """
    user = (await db.scalars(select(models.User).where(models.User.username == username))).first()

    if user is None:
        logger.debug(f"User {username} not found")
        raise ValueError(f"User {username} not found")
    else:
        logger.debug(f"User: {user}")
        return user


async def get_user_by_mail(db: AsyncSession, email: EmailStr) -> models.User:
    """Get a user by email from the DB

    Args:
        db (AsyncSession): Session object
        email (EmailStr): Email of the user to get

    Raises:
        ValueError: If the user is not found

    Returns:
        models.User: User from the database

    """
    user = (await db.scalars(select(models.User).where(models.User.email == email))).first()

    if user is None:
        logger.debug(f"User with email {email} not found")

        raise ValueError(f"User with email {email} not found")
    else:
        logger.debug(f"User: {user}")
        return user


async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[models.User]:
    """Get all users from the DB

    Args:
        db (AsyncSession): Session object
        skip (int, optional): Number of users to skip. Defaults to 0.
        limit (int, optional): Number of users to get. Defaults to 100.

    Raises:
        ValueError: If no users are found in the database

    Returns:
        List[models.User]: List of users
    """
    users = (await db.scalars(select(models.User).offset(skip).limit(limit))).all()

    if users:
        logger.debug(f"Found {len(users)} users in the database.")
        return users
    else:
        raise ValueError("No users found in the database")


async def create_user(db: Session, user: schemas.UserCreate, bg_task: BackgroundTasks):
    """Create new user

    Args:
        user (CreateUser): Create a new user which should be saved in the DB.

    Returns:
        UserInDB: User as represented in the Database.
    """
    hashed_password = await get_password_hash(user.password)
    verification_token = await generate_verification_token()

    # Send email
    body = f"Please verify your email by clicking on the following link: http://192.168.2.135:8000/users/verify_email/{verification_token}"
    bg_task.add_task(send_email, user.email, body, "Verify your email")

    db_user = models.User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        verification_token=verification_token,
    )
    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not create user: {e}")
        raise ValueError("Could not create user")
    else:
        return db_user


async def delete_user(db: AsyncSession, username: str) -> models.User:
    """Delete a user from the database

    Args:
        db (AsyncSession): Database session
        username (str): Username of the user to delete

    Raises:
        ValueError: If the user is not found

    Returns:
        User: User object
    """
    user = await get_user(db, username)

    if user is None:
        raise ValueError(f"User {username} not found")
    
    else:
        try:
            db.delete(user)
            await db.commit()
        except Exception as e:
            logger.error(f"Could not delete user: {e}")
            raise ValueError("Could not delete user")
        else:
            return user


async def verify_email(db: AsyncSession, token: str) -> models.User:
    """Verify the email of the user

    Args:
        db (AsyncSession): Database session
        token (str): Verification token

    Raises:
        ValueError: If the token is invalid

    Returns:
        User: User object
    """
    user = (await db.scalars(select(models.User).where(models.User.verification_token == token))).first()

    if user is None:
        raise ValueError("Invalid verification token")
    
    elif user.email_verified:
        return user
    
    else:
        user.email_verified = True
        user.verification_token = None
        try:
            await db.commit()
            await db.refresh(user)
        except Exception as e:
            logger.error(f"Could not verify email: {e}")
            raise ValueError("Could not verify email")
        else:
            return user
