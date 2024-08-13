"""Create, Update, Delete, Read operations for users in the database."""

import logging
import asyncio
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr

from ...constants import Permission
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
    db: AsyncSession, organization_id: str, user: schemas.UserBase, verify: bool = True
) -> models.Organization:
    """Get an organization by ID from the DB

    Args:
        db (AsyncSession): Session object
        organization_id (str): ID of the organization to get
        verify (bool, optional): Whether to verify if the user is part of the organization. Defaults to True.

    Raises:
        ValueError: If the organization is not found

    Returns:
        models.Organization: Organization from the database
    """

    if verify:
        # Verify if the user is part of the organization
        user = await get_user(db, user.username)
        if not user:
            logger.error(f"User {user.username} not found")
            raise HTTPException(status_code=404, detail="User not found")
        if user.organization_id != organization_id:
            logger.error(f"User {user.username} is not part of the organization")
            raise HTTPException(status_code=401, detail="Unauthorized request")

    organization = (
        await db.scalars(
            select(models.Organization).where(
                models.Organization.organization_id == organization_id
            )
        )
    ).first()

    if organization is None:
        logger.error(f"Organization {organization_id} not found")
        raise HTTPException(status_code=404, detail="Organization not found")
    else:
        logger.debug(f"Organization: {organization}")
        return organization


async def add_permission_to_user(
    db: AsyncSession, user: schemas.UserBase, permission: Permission, resource_id: str
) -> models.ResourcePermission:
    """Add a permission to a user

    Args:
        db (AsyncSession): Async database session
        user (schemas.UserBase): User to add the permission to.
        permission (Permission): Permission type (view or edit)
        resource_id (str): Resource ID to add the permission to.

    Raises:
        HTTPException: If the permission type is invalid.
        ValueError: If the permission cannot be added to the user.

    Returns:
        models.ResourcePermission: Permission object
    """

    permission = models.ResourcePermission(
        user_id=user.username,
        permission_type=str(permission.value),
        resource_id=resource_id,
    )

    try:
        db.add(permission)
        await db.commit()
        await db.refresh(permission)
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not add permission to user: {e}")
        raise ValueError("Could not add permission to user")
    else:
        return permission


async def check_permission(
    db: AsyncSession,
    user: schemas.UserBase,
    resource_id: str,
    permission_type: Permission,
):
    """Check if a user has access to a resource

    Args:
        db (AsyncSession): Database session
        user (schemas.UserBase): User to check the permission for.
        resource_id (str): Resource ID to check the permission for.
        permission_type (Permission): Type of the permission (view or edit)

    Raises:
        HTTPException: If the user does not have the permission

    Returns:
        models.ResourcePermission: Permission object
    """

    user_permission = (
        await db.scalars(
            select(models.ResourcePermission).where(
                and_(
                    models.ResourcePermission.user_id == user.username,
                    models.ResourcePermission.resource_id == resource_id,
                    models.ResourcePermission.permission_type == permission_type.value,
                )
            )
        )
    ).first()

    if user_permission is None:
        logger.error(
            f"User {user.username} does not have permission to {permission_type} resource {resource_id}"
        )
        raise HTTPException(status_code=401, detail="Unauthorized request")
    else:
        logger.debug(
            f"User {user.username} has permission to {permission_type} resource {resource_id}"
        )
        return user_permission


async def create_role(
    db: AsyncSession,
    user: schemas.UserOrganization,
    role_name: str,
) -> models.Role:
    """Create a new role.

    Args:
        db (AsyncSession): Async database session
        user (schemas.UserBase): User creating the role.
        role (schemas.RoleBase): Role to create.

    Returns:
        models.Role: New role
    """
    # Check if the user has edit permission on the organization.
    await check_permission(db, user, user.organization_id, Permission.EDIT)

    try:
        role = models.Role(role_name=role_name)
    except Exception as e:
        logger.error(f"Could not create role: {e}")
        raise ValueError("Could not create role")
    else:
        return role


async def add_role_to_user(
    db: AsyncSession,
    user: schemas.UserOrganization,
    user_id: str,
    role_id: int,
) -> models.UserRole:
    """Add a role to a user in an organization.

    Args:
        db (AsyncSession): _description_
        user (schemas.UserBase): User who adds the role to another user.
        user_id (str): User id of the user to which the role should be added.
        role_id (int): _description_

    Returns:
        models.UserRole: _description_
    """
    # Check if role exists
    role = (
        await db.scalars(select(models.Role).where(models.Role.role_id == role_id))
    ).first()

    if role is None:
        logger.error(f"Role {role_id} not found")
        raise HTTPException(status_code=404, detail="Role not found")

    # Check if the user has edit permission on the organization.
    await check_permission(db, user, user.organization_id, Permission.EDIT)

    user_role = models.UserRole(
        user_id=user_id,
        role_id=role_id,
        organization_id=user.organization_id,
    )

    try:
        db.add(user_role)
        await db.commit()
        await db.refresh(user_role)

    except Exception as e:
        await db.rollback()
        logger.error(f"Could not add role to user: {e}")
        raise ValueError("Could not add role to user")
    else:
        return user_role


async def add_permission_to_role_in_organization(
    db: AsyncSession,
    user: schemas.UserOrganization,
    role_id: int,
    resource_id: str,
    permission_type: Permission,
) -> bool:
    """Add a permission to a role in an organization.

    Args:
        db (AsyncSession): Database session
        user (schemas.UserBase): User adding the permission
        role_id (int): ID of the role
        resource_id (str): ID of the resource
        permission_type (Permission): Type of the permission

    Returns:
        bool: True if successful

    """
    # Check if the user has edit permission on the resource.
    await check_permission(db, user, resource_id, Permission.EDIT)

    user_roles = (
        await db.scalars(
            select(models.UserRole).where(
                and_(
                    models.UserRole.organization_id == user.organization_id,
                    models.UserRole.role_id == role_id,
                )
            )
        )
    ).all()

    tasks = []
    for user_role in user_roles:
        tasks.append(
            add_permission_to_user(db, user_role.user_id, permission_type, resource_id)
        )

    await asyncio.gather(*tasks)
    await db.commit()

    return True


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
        await add_permission_to_user(db, user, Permission.VIEW, organization_id)

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
        # Enable the user to edit the organization
        await add_permission_to_user(db, user, "edit", orga_id)

        return organization


async def delete_organization(
    db: AsyncSession, user: schemas.UserBase, password: str
) -> models.Organization:
    """Delete an organization
    Should also delete all users and resources associated with the organization.

    Args:
        db (AsyncSession): Database session
        user (schemas.UserBase): User deleting the organization
        password (str): Password of the user

    Raises:
        HTTPException: If the user is not authorized to delete the organization
        HTTPException: If the organization is not found

    Returns:
        models.Organization: Organization object
    """

    # TODO: Check if this is actually correct and all records are deleted

    user = await get_user(db, user.username)

    if not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Unauthorized request")

    check_permission(db, user, user.organization_id, Permission.EDIT)

    try:
        organization = await get_organization(db, user.organization_id, user, False)
    except ValueError:
        raise HTTPException(status_code=404, detail="Organization not found")

    try:
        db.delete(organization)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not delete organization: {e}")
        raise ValueError("Could not delete organization")
    else:
        return organization


async def remove_user_organizaion(
    db: AsyncSession, user: schemas.UserOrganization, user_id
) -> models.User:
    """Remove a user from an organization.
    This should also remove all privileges and permissions of the user in the organization.

    Args:
        db (AsyncSession): Database session
        user (schemas.UserOrganization): User removing the user from the organization
        user_id (str): ID of the user to remove

    Raises:
        HTTPException: If the user is not found
        HTTPException: If the user is not part of the organization
        HTTPException: If the user tries to remove themselves from the organization

    Returns:
        models.User: User object
    """

    # TODO: Check if all permissions etc are removed properly.
    check_permission(db, user, user.organization_id, Permission.EDIT)

    target_user = await get_user(db, user_id)

    if target_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if target_user.organization_id != user.organization_id:
        raise HTTPException(status_code=401, detail="Unauthorized request")

    if target_user.username == user.username:
        raise HTTPException(
            status_code=400, detail="Cannot remove yourself from the organization"
        )

    target_user.organization_id = None

    try:
        db.add(target_user)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Could not remove user from organization: {e}")
        raise ValueError("Could not remove user from organization")
    else:
        # Add seat back to the organization
        organization = await get_organization(db, user.organization_id, user, False)
        organization.seats += 1
        db.add(organization)
        await db.commit()

        return True
