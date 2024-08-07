import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import HTTPException

from ..schemas import UserBase, TFInDB
from ..models import TFModels

logger = logging.getLogger("CRUD Models")


async def get_model(model_id: str, db: AsyncSession, user: UserBase) -> TFInDB:
    """Get a model by its ID.

    Args:
        model_id (str): Model ID.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the model is not found or the user is unauthorized.

    Returns:
        TFInDB: Model object.
    """

    model = (
        await db.scalars(
            select(TFModels)
            .where(TFModels.tf_id == model_id)
            .order_by(TFModels.version.desc())
        )
    ).first()

    if model is None:
        logger.error(f"Model {model_id} not found.")
        raise HTTPException(status_code=404, detail="Model not found.")
    elif model.user_id != user.username:
        logger.error(f"Unauthorized request for model {model_id} from {user.username}.")
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return model


async def get_model_versions(
    model_id: str, db: AsyncSession, user: UserBase
) -> list[TFInDB]:
    """Get all versions of a model.

    Args:
        model_id (str): Model ID.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the model is not found or the user is unauthorized.

    Returns:
        list[TFInDB]: List of model objects.
    """
    models = (
        await db.scalars(select(TFModels).where(TFModels.tf_id == model_id))
    ).all()
    if not models:
        logger.error(f"Model {model_id} not found.")
        raise HTTPException(status_code=404, detail="Model not found.")
    elif models[0].user_id != user.username:
        logger.error(f"Unauthorized request for model {model_id} from {user.username}.")
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return models


async def get_model_version(
    model_id: str, version: int, db: AsyncSession, user: UserBase
) -> TFInDB:
    """Get a specific version of a model.

    Args:
        model_id (str): Model ID.
        version (int): Model version.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the model is not found or the user is unauthorized.

    Returns:
        TFInDB: Model object.
    """
    model = (
        await db.scalars(
            select(TFModels).where(
                and_(TFModels.tf_id == model_id, TFModels.version == version)
            )
        )
    ).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    elif model.user_id != user.username:
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return model


async def save_model_to_db(
    model_id: str,
    model_path: str | Path,
    user: UserBase,
    db: AsyncSession,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Save the model to the database.

    Args:
        model_id (str): Model ID.
        model_path (str): Path to the model.
        user (UserBase): Current user.
        db (AsyncSession): Database session.
        model_name (str, Optional): Model name.
        description (str, Optional): Model description.

    Returns:
        TFInDB: Model object.
    """

    try:
        model_in_db = await get_model(model_id, db, user)
    except HTTPException:
        logger.debug(f"Model {model_id} not found. Creating a new model.")
        model_in_db = TFModels(
            model_id=model_id,
            model_name=model_name,
            description=description,
            user_id=user.username,
            version=1,
            model_path=str(model_path),
        )

    else:
        logger.debug(f"Updating model {model_id}.")
        model_in_db.version += 1
        model_in_db.model_path = str(model_path)
        model_in_db.model_name = model_name
        model_in_db.description = description

    db.add(model_in_db)
    await db.commit()
    return model_in_db


async def get_models(user: UserBase, db: AsyncSession) -> list[TFInDB]:
    """Get all models for the current user.

    Args:
        user (UserBase): Current user.
        db (AsyncSession): Database session.

    Returns:
        list[TFInDB]: List of model objects.
    """

    subquery = (
        select(TFModels.tf_id, TFModels.version)
        .group_by(TFModels.tf_id)
        .having(TFModels.version == select(func.max(TFModels.version)))
    ).subquery()

    query = (
        select(TFModels)
        .join(
            subquery,
            and_(
                TFModels.tf_id == subquery.c.tf_id,
                TFModels.version == subquery.c.version,
            ),
        )
        .where(TFModels.user_id == user.username)
    )

    return (await db.scalars(query)).all()
