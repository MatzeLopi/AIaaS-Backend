import logging
from asyncio import run
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import Json

from ..constants import LOGLEVEL
from ..logic import tf_models
from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..backend import dependencies
from ..backend.crud import models_crud

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger("Models Router")

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)

# Not run as async since the create model is slow and event loop should not be blocked
@router.post("/create_model")
def create_model(
    current_user: USER_DEPENDENCY,
    model_definition: Json,
    model_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Create a tensorflow Deep learning model."""
    model_path: str = tf_models.create_model(model_definition)
    run(
        models_crud.save_model_to_db(
            model_path, current_user, db, model_name=model_name
        )
    )

    return HTTPException(status_code=201, detail="Model created.")


@router.get("/models")
async def get_models(
    current_user: USER_DEPENDENCY,
    db: AsyncSession = Depends(get_db),
):
    """Get all models for the current user."""
    try:
        models = await models_crud.get_models(current_user, db)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return HTTPException(status_code=500, detail="Error getting models.")
    else:
        return models 


@router.get("/layers")
async def get_layers():
    """Get all available layers."""
    return tf_models.get_layers()


@router.get("/model/versions")
async def get_model_versions(
    current_user: USER_DEPENDENCY,
    model_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all versions of a model."""
    return await models_crud.get_model_versions(model_name, db, current_user)


@router.get("/model/latest")
async def get_model(
    current_user: USER_DEPENDENCY, model_name: str, db: AsyncSession = Depends(get_db)
):
    """Get a model by its ID."""
    return models_crud.get_model(model_name, db, current_user)


@router.get("/model")
async def get_model_by_version(
    current_user: USER_DEPENDENCY,
    model_name: str,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific version of a model.
    
    Args:
        current_user (USER_DEPENDENCY): Current user.
        model_name (str): Model name.
        version (int): Model version.
        db (AsyncSession, optional): Database session. Defaults to Depends(get_db).
    
    Returns:
        TFInDB: Model object.
    """
    return await models_crud.get_model_version(model_name, version, db, current_user)
