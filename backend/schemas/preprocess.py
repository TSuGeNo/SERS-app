"""Preprocessing schemas"""

from pydantic import BaseModel
from typing import Optional, List

class PreprocessingOptions(BaseModel):
    baseline_correction: bool = True
    baseline_lambda: float = 1e5
    baseline_p: float = 0.01
    smoothing: bool = True
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    normalization: str = 'vector'
    peak_detection: bool = True
    peak_prominence: float = 0.1
    peak_distance: int = 10

class PreprocessRequest(BaseModel):
    wavenumber: List[float]
    intensity: List[float]
    options: Optional[PreprocessingOptions] = None

class PeakInfo(BaseModel):
    wavenumber: float
    intensity: float
    prominence: float
    width: float

class PreprocessResponse(BaseModel):
    success: bool
    wavenumber: List[float]
    original_intensity: List[float]
    processed_intensity: List[float]
    peaks: List[PeakInfo]
    applied_steps: List[str]
