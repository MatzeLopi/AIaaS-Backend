

# TODO: Add function to save model name and owner to the database
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import UserBase
from ..models import ModelTable

async def save_model(model_path:str, user:UserBase, db:AsyncSession):
    """Save the model to the database."""
    pass

async def get_models(user:UserBase, db:AsyncSession):
    """Get all models for the current user."""
    pass