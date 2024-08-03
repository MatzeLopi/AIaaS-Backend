""" CRUD file for Datasets. """

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def save_dataset_to_db(
    db: AsyncSession,
    dataset_id,
    user: schemas.UserBase,
    dataset_path: Path,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    dataset_type: Optional[str] = None,
):
    """Save a dataset to the database."""
    pass


async def get_available_datasets(db: AsyncSession, user: schemas.UserBase):
    """Get all available datasets.

    Args:
        db (AsyncSession): Session object
        user (schemas.UserBase): User object.

    Returns:
        list[models.UserTable]: List of available datasets.
    """
    return (
        await db.scalars(
            select(models.UserTable)
            .where(models.UserTable.username == user.username)
            .order_by(models.UserTable.created_at.desc())
        )
    ).all()
