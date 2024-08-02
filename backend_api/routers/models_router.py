from asyncio import run
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.ext.asyncio import AsyncSession

from ..logic import tf_models
from ..backend.dependencies import get_db
from ..backend import schemas, dependencies
from ..backend.crud import models_crud



router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(dependencies.get_current_active_user)])

@router.post("/model")
def create_model(
    model: str,
    db: AsyncSession = Depends(get_db),
    current_user: schemas.UserBase = Depends(dependencies.get_current_active_user),
):
    """Create a tensorflow Deep learning model."""
    model_path:str = tf_models.create_model(model)

    run(models_crud.save_model(model_path, current_user, db))

    return {"message": "Model created successfully."}


@router.get("/models")
async def get_models(
    db: AsyncSession = Depends(get_db),
    current_user: schemas.UserBase = Depends(dependencies.get_current_active_user),
):
    """Get all models for the current user."""
    return await models_crud.get_models(current_user, db)


@router.get("layers")
async def get_layers():
    """Get all available layers."""
    return tf_models.get_layers()