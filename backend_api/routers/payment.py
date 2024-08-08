"""Router for payment endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..backend.dependencies import get_db, USER_DEPENDENCY
from ..backend import dependencies

router = APIRouter(
    prefix="/payment",
    tags=["payment"],
    dependencies=[Depends(dependencies.get_current_active_user)],
)


@router.post("/payment/add_cpu")
async def add_cpu_payment():
    pass


@router.post("/payment/add_gpu")
async def add_gpu_payment():
    pass


@router.post("/payment/add_seat")
async def add_seat_payment():
    pass


@router.get("/payment/history")
async def get_payment_history():
    pass
