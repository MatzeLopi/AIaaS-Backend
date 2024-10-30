""" Router for organization endpoints."""
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends, HTTPException, status
from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..backend import dependencies

from ..backend.crud import organizations

router = APIRouter(
    prefix="/organizations",
    tags=["organizations"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)


@router.post("/organization/new")
async def new_organization(name:str, id:str, user:USER_DEPENDENCY, db:AsyncSession = Depends(get_db)):
    await organizations.create_organization(organization_id=id, organization_name=name, user=user, db=db)


@router.post("/organization/invite")
async def invite_user():
    pass


@router.get("/organization/members")
async def get_organization_members():
    pass


@router.get("/organization/me")
async def get_my_organization():
    pass
