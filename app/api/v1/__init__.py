from api.v1.models.router import router as models_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(models_router)
