"""v1 API router — aggregates sub-routers for each resource."""
from fastapi import APIRouter

from app.api.v1.transactions import router as transactions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(transactions_router)

__all__ = ["api_router"]
