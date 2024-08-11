"""Create, Update, Delete, Read operations for users in the database."""

import logging
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException
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
    user = (
        await db.scalars(select(models.User).where(models.User.username == username))
    ).first()

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
    user = (
        await db.scalars(select(models.User).where(models.User.email == email))
    ).first()

    if user is None:
        logger.debug(f"User with email {email} not found")

        raise ValueError(f"User with email {email} not found")
    else:
        logger.debug(f"User: {user}")
        return user


async def get_users(
    db: AsyncSession, skip: int = 0, limit: int = 100
) -> list[models.User]:
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
    user = (
        await db.scalars(
            select(models.User).where(models.User.verification_token == token)
        )
    ).first()

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


async def get_organization(
    db: AsyncSession, organization_id: str
) -> models.Organization:
    """Get an organization by ID from the DB

    Args:
        db (AsyncSession): Session object
        organization_id (str): ID of the organization to get

    Raises:
        ValueError: If the organization is not found

    Returns:
        models.Organization: Organization from the database
    """
    organization = (
        await db.scalars(
            select(models.Organization).where(
                models.Organization.organization_id == organization_id
            )
        )
    ).first()

    if organization is None:
        logger.error(f"Organization {organization_id} not found")
        raise ValueError(f"Organization {organization_id} not found")
    else:
        logger.debug(f"Organization: {organization}")
        return organization


async def add_permission_to_user(
    db: AsyncSession, user: schemas.UserBase, permission_id: int
):
    pass


async def check_permission(
    db: AsyncSession, user: schemas.UserBase, resource_id: int, permission_type: str
):
    pass


async def add_role_to_user(db: AsyncSession, user: schemas.UserBase, role_id: int):
    pass


async def add_user_organization(
    db: AsyncSession,
    user: schemas.UserBase,
    organization_id: str,
    overwrite_seats: bool = False,
) -> models.Organization:
    """Add a user to an organization

    Args:
        db (AsyncSession): Database session
        user (schemas.UserBase): User to add to the organization
        organization_id (str): ID of the organization
        overwrite_seats (bool, optional): Whether to ignore if the organization has seats left. Defaults to False.

    Raises:
        ValueError: If the user or organization is not found
        HTTPException: If there are no seats left in the organization

    Returns:
        models.Organization: Organization object
    """
    # Update the foreign key in the user_roles table
    user = await get_user(db, user.username)
    organization = await get_organization(db, organization_id)

    if organization.seats <= 0 and not overwrite_seats:
        HTTPException(status_code=400, detail="No seats left in the organization")

    user.organization_id = organization_id

    if not overwrite_seats:
        organization.seats -= 1
    try:
        db.add(user)
        db.add(organization)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not add user to organization: {e}")
        raise ValueError("Could not add user to organization")
    else:
        return organization


async def create_organization(
    db: AsyncSession, organization: schemas.OrganizationBase, user: schemas.UserBase
) -> models.Organization:
    """Create a new organization

    Args:
        db (AsyncSession): Database session
        organization (schemas.OrganizationBase): Organization to create
        user (schemas.UserBase): User creating the organization

    Raises:
        HTTPException: If the organization cannot be created

    Returns:
        models.Organization: Organization object
    """
    orga_id = f"orga_{uuid4().hex}"
    organization = models.Organization(
        organization_name=organization.organization_name,
        organization_description=organization.organization_description,
        organization_id=orga_id,
        seats=0,
        cpu_hours=0,
        gpu_hours=0,
    )

    try:
        db.add(organization)
        await db.commit()
        await db.refresh(organization)
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not create organization: {e}")
        raise HTTPException(status_code=500, detail="Could not create organization")
    else:
        # Add current user to the organization
        await add_user_organization(db, user, orga_id, True)

        return organization


async def delete_organization(
    db: AsyncSession, organization_id: int
) -> models.Organization:
    pass


async def remove_user_organizaion():
    pass
