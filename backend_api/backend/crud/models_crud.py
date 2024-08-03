# TODO: Add function to save model name and owner to the database
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import UserBase, TFInDB
from ..models import TFModels


async def save_model_to_db(
    model_id: str, model_path: str, user: UserBase, db: AsyncSession
):
    """Save the model to the database."""
    pass


async def get_model_location(model_id: str, user: UserBase, db: AsyncSession) -> str:
    """Get all models for the current user."""
    pass
