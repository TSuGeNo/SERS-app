"""Simulation schemas"""

from pydantic import BaseModel
from typing import List, Optional

class SimulateRequest(BaseModel):
    material: str = 'Ag'
    nanoparticle_size: float = 50
    shape: str = 'sphere'
    excitation_wavelength: float = 785

class MaterialProperties(BaseModel):
    name: str
    epsilon_inf: float
    omega_p: float
    gamma: float

class SimulateResponse(BaseModel):
    success: bool
    lspr_peak: float
    enhancement_factor: str
    fwhm: float
    wavelengths: List[float]
    spectrum: List[float]
    recommendation: str
    material_properties: MaterialProperties
