import logging
from asyncio import run
from typing import Optional

from fastapi import APIRouter, Depends
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


@router.post("/model")
def create_model(
    current_user: USER_DEPENDENCY,
    model_definition: Json,
    model_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Create a tensorflow Deep learning model."""
    model_path: str = tf_models.create_model(model_definition)

    run(
        models_crud.save_model_to_db(
            model_path, current_user, db, model_name=model_name
        )
    )

    return {"message": "Model created successfully."}


@router.get("/models")
async def get_models(
    current_user: USER_DEPENDENCY,
    db: AsyncSession = Depends(get_db),
):
    """Get all models for the current user."""
    return await models_crud.get_models(current_user, db)


@router.get("layers")
async def get_layers():
    """Get all available layers."""
    return tf_models.get_layers()


@router.get("/model/versions")
async def get_model_versions(
    current_user: USER_DEPENDENCY,
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all versions of a model."""
    return await models_crud.get_model_versions(model_id, db, current_user)


@router.get("/model/latest")
async def get_model(
    current_user: USER_DEPENDENCY, model_id: str, db: AsyncSession = Depends(get_db)
):
    """Get a model by its ID."""
    return models_crud.get_model(model_id, db, current_user)


@router.get("/model")
async def get_model_by_version(
    current_user: USER_DEPENDENCY,
    model_id: str,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific version of a model."""
    return await models_crud.get_model_version(model_id, version, db, current_user)
