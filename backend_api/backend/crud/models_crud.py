import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import HTTPException

from ..schemas import UserBase, TFInDB
from ..models import TFModel

logger = logging.getLogger("CRUD Models")


async def get_model(model_name: str, db: AsyncSession, user: UserBase) -> TFInDB:
    """Get a model by its name.

    Args:
        model_id (str): Name of the model.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the model is not found or the user is unauthorized.

    Returns:
        TFInDB: Model object.
    """

    model = (
        await db.scalars(
            select(TFModel)
            .where(TFModel.tf_name == model_name)
            .order_by(TFModel.version.desc())
        )
    ).first()

    if model is None:
        logger.error(f"Model {model_name} not found.")
        raise HTTPException(status_code=404, detail="Model not found.")
    elif model.user_id != user.user_id:
        logger.error(f"Unauthorized request for model {model_name} from {user.user_id}.")
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return model


async def get_model_versions(
    model_name: str, db: AsyncSession, user: UserBase
) -> list[TFInDB]:
    """Get all versions of a model.

    Args:
        model_id (str): Model name.
        db (AsyncSession): Database session.
        user (UserBase): Current user.

    Raises:
        HTTPException: Raised if the model is not found or the user is unauthorized.

    Returns:
        list[TFInDB]: List of model objects.
    """
    models = (
        await db.scalars(select(TFModel).where(TFModel.tf_name == model_name))
    ).all()
    if not models:
        logger.error(f"Model {model_name} not found.")
        raise HTTPException(status_code=404, detail="Model not found.")
    elif models[0].user_id != user.user_id:
        logger.error(f"Unauthorized request for model {model_name} from {user.user_id}.")
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return models


async def get_model_version(
    model_name: str, version: int, db: AsyncSession, user: UserBase
) -> TFInDB:
    """Get a specific version of a model.

    Args:
        model_name (str): Name of the model
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
            select(TFModel).where(
                and_(TFModel.tf_name == model_name, TFModel.version == version)
            )
        )
    ).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    elif model.user_id != user.user_id:
        logger.error(f"Unauthorized request for model {model_name} from {user.user_id}.")
        raise HTTPException(status_code=401, detail="Unauthorized request.")
    else:
        return model


async def save_model_to_db(
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
    model_id = uuid4().hex
    try:
        model_in_db = await get_model(model_name, db, user)
    except HTTPException:
        logger.debug(f"Model {model_id} not found. Creating a new model.")
        model_in_db = TFModel(
            model_id=model_id,
            model_name=model_name,
            description=description,
            user_id=user.user_id,
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
        select(TFModel.tf_id, TFModel.version)
        .group_by(TFModel.tf_id)
        .having(TFModel.version == select(func.max(TFModel.version)))
    ).subquery()
    query = (
        select(TFModel)
        .join(
            subquery,
            and_(
                TFModel.tf_id == subquery.c.tf_id,
                TFModel.version == subquery.c.version,
            ),
        )
        .where(TFModel.user_id == user.user_id)
    )

    return (await db.scalars(query)).all()
