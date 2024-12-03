import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import HTTPException

from ..schemas import UserBase, OrganizationBase
from ..models import Organization

logger = logging.getLogger("CRUD Organizations")


async def check_name(organization_id: str, db: AsyncSession):
    """Check if the organization name is already taken.

    Args:
        organization_id (str): Organization ID.
        db (AsyncSession): Database session.

    Returns:
        bool: True if the organization exists, False otherwise.
    """

    organization = await db.scalars(
        select(OrganizationBase).where(
            OrganizationBase.organization_id == organization_id
        )
    )
    if organization is None:
        return False
    else:
        return True


async def get_organization(
    organization_id: str, db: AsyncSession, user: UserBase
) -> OrganizationBase:
    """Get an organization by its ID.

    Args:
        organization_id (str): Organization ID.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the organization is not found or the user is unauthorized.

    Returns:
        OrganizationBase: Organization object.
    """

    organization = (
        await db.scalars(
            select(Organization)
            .where(Organization.organization_id == organization_id)
            .order_by(Organization.version.desc())
        )
    ).first()

    if organization is None:
        logger.error(f"Organization {organization_id} not found.")
        raise HTTPException(status_code=404, detail="Organization not found.")
    else:
        return organization


async def create_organization(organization_id: str, db: AsyncSession, user: UserBase, organization_name:str | None = None) -> OrganizationBase:
    """ Create a new organization.

    Args:
        organization_id (str): Organization ID.
        db (AsyncSession): Database session.
        user (UserBase): Current user.
        organization_name (str | None, optional): Organization name. Defaults to None.

    Raises:
        HTTPException: If the organization already exists.
        HTTPException: Organization could not be created.

    Returns:
        OrganizationBase: Organization object.
    """
    # Check if the organization already exists
    if await check_name(organization_id, db):
        logger.error(f"Organization {organization_id} already exists.")
        raise HTTPException(status_code=400, detail="Organization already exists.")

    try:
        organization = Organization(
            organization_id=organization_id,
            user_id=user.user_id,
            seats=1,
            gpu_hours=0,
            cpu_hours=0,
        )
    except Exception as e:
        logger.error(f"Could not create organization: {e}")
        raise HTTPException(status_code=400, detail="Could not create organization")
    else:
        return organization

async def get_organization_me(user: UserBase, db: AsyncSession) -> OrganizationBase:
    """ Get the organization of the current user.

    Args:
        user (UserBase): Current user.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If the organization is not found.
    
    Returns:
        OrganizationBase: Organization object.
    """
    organization = (
        await db.scalars(
            select(Organization)
            .where(Organization.organization_id == user.organization_id)
        )
    ).first()

    if organization is None:
        logger.error(f"Organization for user {user.user_id} not found.")
        raise HTTPException(status_code=404, detail="Organization not found.")
    else:
        return organization