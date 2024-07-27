"""Create, Update, Delete, Read operations for users in the database."""

import logging

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from pydantic import EmailStr

from .. import models, schemas
from ..utils import get_password_hash, generate_verification_token, send_email, retry
from ...constants import DBError 

logger = logging.getLogger(__name__)

@retry((OperationalError))
def get_user(db: Session, username: str) -> models.User:
    """Get a user by name from the DB

    Args:
        db (Session): Session object
        username (str): Name of the user to get

    Raises:
        ValueError: If the user is not found

    Returns:
        models.User: User from the database
    """
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        logger.debug(f"User {username} not found")
        raise ValueError(f"User {username} not found")
    else:
        logger.debug(f"User: {user}")
        return user


def get_user_by_mail(db: Session, email: EmailStr) -> models.User:
    """Get a user by email from the DB

    Args:
        db (Session): Session object
        email (EmailStr): Email of the user to get

    Raises:
        ValueError: If the user is not found

    Returns:
        models.User: User from the database

    """
    user = db.query(models.User).filter(models.User.email == email).first()

    if user is None:
        logger.debug(f"User with email {email} not found")
        raise ValueError(f"User with email {email} not found")
    else:
        logger.debug(f"User: {user}")
        return user


def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[models.User]:
    """Get all users from the DB

    Args:
        db (Session): Session object
        skip (int, optional): Number of users to skip. Defaults to 0.
        limit (int, optional): Number of users to get. Defaults to 100.

    Raises:
        ValueError: If no users are found in the database

    Returns:
        List[models.User]: List of users
    """
    users = db.query(models.User).offset(skip).limit(limit).all()

    if users:
        logger.debug(f"Found {len(users)} users in the database.")
        return users
    else:
        raise ValueError("No users found in the database")


def create_user(db: Session, user: schemas.UserCreate, bg_task: BackgroundTasks):
    """Create new user

    Args:
        user (CreateUser): Create a new user which should be saved in the DB.

    Returns:
        UserInDB: User as represented in the Database.
    """
    hashed_password = get_password_hash(user.password)
    verification_token = generate_verification_token()

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
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        logger.error(f"Could not create user: {e}")
        raise DBError("Could not create user in the database")
    else:
        return db_user


def delete_user(db: Session, username: str) -> models.User:
    """Delete a user from the database

    Args:
        db (Session): Database session
        username (str): Username of the user to delete

    Raises:
        ValueError: If the user is not found

    Returns:
        User: User object
    """
    user = db.query(models.User).filter(models.User.username == username).first()

    if user is None:
        raise ValueError(f"User {username} not found")
    else:
        try:
            db.delete(user)
            db.commit()
        except Exception as e:
            logger.error(f"Could not delete user: {e}")
            raise DBError("Could not delete user from the database")
        else:
            return user


def verify_email(db: Session, token: str) -> models.User:
    """Verify the email of the user

    Args:
        db (Session): Database session
        token (str): Verification token

    Raises:
        ValueError: If the token is invalid

    Returns:
        User: User object
    """
    user = db.query(models.User).filter(models.User.verification_token == token).first()

    if user is None:
        raise ValueError("Invalid verification token")
    elif user.email_verified:
        return user
    else:
        user.email_verified = True
        user.verification_token = None
        try:
            db.commit()
            db.refresh(user)
        except Exception as e:
            logger.error(f"Could not verify email: {e}")
            raise DBError("Could not verify email")
        else:
            return user
