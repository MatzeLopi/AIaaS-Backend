""" CRUD file for Datasets. """

import logging
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, schemas

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def get_dataset(db: AsyncSession, dataset_id: str, user: schemas.UserBase):
    """Get a dataset by its ID.

    Args:
        db (AsyncSession): Database session.
        dataset_id (str): Dataset ID.
        user (schemas.UserBase): Current user.

    Returns:
        models.Datasets: Dataset object.
    """
    dataset = (
        await db.scalars(
            select(models.Datasets)
            .where(models.Datasets.dataset_id == dataset_id)
            .order_by(models.Datasets.version.desc())
        )
    ).first()

    if dataset is None:
        logger.error(f"Dataset {dataset_id} not found.")
        raise HTTPException(status_code=404, detail="Dataset not found.")
    elif dataset.user_id != user.user_id:
        logger.error(
            f"Unauthorized request for dataset {dataset_id} from {user.user_id}."
        )
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return dataset


async def save_dataset_to_db(
    db: AsyncSession,
    dataset_id,
    user: schemas.UserBase,
    dataset_path: Path,
    dataset_type: str,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Save a dataset to the database.

    Args:
        db (AsyncSession): Database session.
        dataset_id (str): Dataset ID.
        user (schemas.UserBase): Current user.
        dataset_path (Path): Path to the dataset.
        dataset_type (str): Type of the dataset.
        dataset_name (str, Optional): Dataset name.
        description (str, Optional): Dataset description.
    """

    try:
        dataset_in_db = await get_dataset(db, dataset_id, user)
    except HTTPException:
        # Create a new dataset
        dataset_in_db = models.Datasets(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_description=description,
            user_id=user.user_id,
            dataset_type=dataset_type,
            version=1,
            dataset_path=str(dataset_path),
        )

    else:
        # Update the existing dataset
        dataset_in_db.version += 1
        dataset_in_db.dataset_path = str(dataset_path)
        dataset_in_db.dataset_name = dataset_name
        dataset_in_db.dataset_description = description

    db.add(dataset_in_db)
    await db.commit()
    return dataset_in_db


async def get_available_datasets(db: AsyncSession, user: schemas.UserBase):
    """Get all available datasets with unique id.

    Args:
        db (AsyncSession): Session object
        user (schemas.UserBase): User object.

    Returns:
        list[models.UserTable]: List of available datasets.
    """

    subquery = (
        select(models.Datasets.dataset_id, models.Datasets.version)
        .group_by(models.Datasets.dataset_id)
        .having(
            models.Datasets.version
            == select(func.max(models.Datasets.version)).where(
                models.Datasets.dataset_id == models.Datasets.dataset_id
            )
        )
    ).subquery()

    query = (
        select(models.Datasets)
        .join(
            subquery,
            and_(
                models.Datasets.dataset_id == subquery.c.dataset_id,
                models.Datasets.version == subquery.c.version,
            ),
        )
        .where(models.Datasets.user_id == user.user_id)
    )

    datasets = (await db.execute(query)).all()
    logger.debug(f"Datasets: {datasets}")
    return datasets


async def get_dataset_version(
    db: AsyncSession, dataset_id: str, version: int, user: schemas.UserBase
):
    """Get a specific version of a dataset.

    Args:
        db (AsyncSession): Database session.
        dataset_id (str): Dataset ID.
        version (int): Version number.
        user (schemas.UserBase): Current user.

    Returns:
        models.Datasets: Dataset object.
    """
    dataset = (
        await db.scalars(
            select(models.Datasets).where(
                and_(
                    models.Datasets.dataset_id == dataset_id,
                    models.Datasets.version == version,
                )
            )
        )
    ).first()

    if dataset is None:
        logger.error(f"Dataset {dataset_id} not found.")
        raise HTTPException(status_code=404, detail="Dataset not found.")
    elif dataset.user_id != user.user_id:
        logger.error(
            f"Unauthorized request for dataset {dataset_id} from {user.user_id}."
        )
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return dataset


async def get_dataset_versions(db: AsyncSession, dataset_id, user: schemas.UserBase):
    """Get all versions of a dataset.

    Args:
        db (AsyncSession): Database session.
        dataset_id (str): Dataset ID.
        user (schemas.UserBase): Current user.

    Returns:
        list[models.Datasets]: List of dataset objects.
    """
    dataset = (
        await db.scalars(
            select(models.Datasets).where(models.Datasets.dataset_id == dataset_id)
        )
    ).all()

    if not dataset:
        logger.error(f"Dataset {dataset_id} not found.")
        raise HTTPException(status_code=404, detail="Dataset not found.")
    elif dataset[0].user_id != user.user_id:
        logger.error(
            f"Unauthorized request for dataset {dataset_id} from {user.user_id}."
        )
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return dataset
