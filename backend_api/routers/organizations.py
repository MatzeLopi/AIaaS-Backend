""" Router for organization endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..backend import dependencies

router = APIRouter(
    prefix="/organizations",
    tags=["organizations"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)


@router.post("/organization/new")
async def new_organization():
    pass


@router.post("/organization/invite")
async def invite_user():
    pass


@router.get("/organization/members")
async def get_organization_members():
    pass


@router.get("/organization/me")
async def get_my_organization():
    pass
