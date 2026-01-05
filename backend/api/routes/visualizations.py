"""Visualization endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
import numpy as np

router = APIRouter()

@router.get("/visualizations/types")
async def get_visualization_types():
    """Get available visualization types"""
    return {
        "types": [
            {"id": "spectrum", "name": "SERS Spectrum", "description": "Line plot with peaks"},
            {"id": "pca", "name": "PCA Scatter", "description": "2D/3D PCA plot"},
            {"id": "heatmap", "name": "Heatmap", "description": "Intensity matrix"},
            {"id": "confusion", "name": "Confusion Matrix", "description": "Classification results"},
        ]
    }

@router.post("/visualizations/generate")
async def generate_visualization(viz_type: str, data: Optional[dict] = None):
    """Generate a visualization"""
    return {"success": True, "type": viz_type, "url": f"/static/viz_{viz_type}.png"}

@router.get("/visualizations/{viz_id}")
async def get_visualization(viz_id: str):
    """Get a specific visualization"""
    return {"id": viz_id, "type": "spectrum", "data": {}}
