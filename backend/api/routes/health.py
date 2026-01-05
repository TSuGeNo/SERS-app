"""Health check endpoints"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sers-insight-api",
    }


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for load balancers"""
    # Add database/redis connection checks here
    return {
        "status": "ready",
        "checks": {
            "database": "ok",
            "redis": "ok",
        }
    }
