"""LSPR Simulation endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np

from schemas.simulate import SimulateRequest, SimulateResponse, MaterialProperties

router = APIRouter()

# Drude-Lorentz model parameters for Ag and Au
MATERIAL_PARAMS = {
    'Ag': {
        'name': 'Silver',
        'epsilon_inf': 5.0,  # High-frequency dielectric constant
        'omega_p': 9.17,  # Plasma frequency (eV)
        'gamma': 0.021,  # Damping constant (eV)
        'lspr_peak_base': 400,  # Base LSPR peak wavelength (nm)
    },
    'Au': {
        'name': 'Gold',
        'epsilon_inf': 9.5,
        'omega_p': 9.02,
        'gamma': 0.071,
        'lspr_peak_base': 520,
    },
}

# Shape correction factors
SHAPE_FACTORS = {
    'sphere': {'peak_shift': 0, 'ef_factor': 1.0},
    'rod': {'peak_shift': 50, 'ef_factor': 1.5},  # Longitudinal mode red-shifted
    'star': {'peak_shift': 80, 'ef_factor': 2.0},  # Hot spots increase EF
    'cube': {'peak_shift': 20, 'ef_factor': 1.2},
}


def calculate_drude_dielectric(wavelength_nm: float, material: str) -> complex:
    """
    Calculate dielectric function using Drude-Lorentz model.
    
    ε(ω) = ε_∞ - ω_p² / (ω² + iγω)
    """
    params = MATERIAL_PARAMS[material]
    
    # Convert wavelength to energy (eV)
    h = 4.136e-15  # Planck constant in eV·s
    c = 3e17  # Speed of light in nm/s
    omega = h * c / wavelength_nm  # Energy in eV
    
    epsilon_inf = params['epsilon_inf']
    omega_p = params['omega_p']
    gamma = params['gamma']
    
    # Drude model
    epsilon = epsilon_inf - omega_p**2 / (omega**2 + 1j * gamma * omega)
    
    return epsilon


def mie_enhancement_factor(size_nm: float, material: str, wavelength_nm: float) -> float:
    """
    Estimate SERS enhancement factor using simplified Mie theory.
    
    EF ∝ |E_local/E_0|⁴ ∝ |α|⁴ / V²
    
    where α is the polarizability
    """
    epsilon = calculate_drude_dielectric(wavelength_nm, material)
    epsilon_m = 1.33**2  # Water medium
    
    # Polarizability for sphere (simplified)
    alpha = 4 * np.pi * (size_nm/2)**3 * (epsilon - epsilon_m) / (epsilon + 2 * epsilon_m)
    
    # Enhancement factor (simplified estimate)
    ef_value = abs(alpha)**4 / ((size_nm/2)**6 * 1e6)
    ef_exponent = np.log10(max(ef_value, 1))
    
    return min(ef_exponent, 12)  # Cap at 10^12


def calculate_lspr_peak(material: str, size_nm: float, shape: str) -> float:
    """
    Calculate LSPR peak wavelength based on material, size, and shape.
    
    Accounts for size-dependent red shift and shape effects.
    """
    base_peak = MATERIAL_PARAMS[material]['lspr_peak_base']
    shape_shift = SHAPE_FACTORS[shape]['peak_shift']
    
    # Size-dependent shift (larger particles → red shift)
    size_shift = (size_nm - 50) * 0.5  # ~0.5 nm shift per nm size increase
    
    return base_peak + shape_shift + size_shift


def calculate_fwhm(material: str, size_nm: float) -> float:
    """Calculate Full Width at Half Maximum of LSPR peak"""
    base_fwhm = 40 if material == 'Ag' else 60  # Ag has sharper peaks
    size_contribution = (size_nm - 50) * 0.3  # Broader with size
    
    return base_fwhm + size_contribution


@router.post("/simulate", response_model=SimulateResponse)
async def run_simulation(request: SimulateRequest):
    """
    Run LSPR simulation for nanoparticle configuration.
    
    Calculates:
    - LSPR peak wavelength
    - Enhancement factor estimate
    - FWHM (Full Width at Half Maximum)
    - Recommendations for optimal configuration
    """
    try:
        material = request.material
        size = request.nanoparticle_size
        shape = request.shape
        excitation = request.excitation_wavelength
        
        if material not in MATERIAL_PARAMS:
            raise HTTPException(status_code=400, detail=f"Unknown material: {material}")
        if shape not in SHAPE_FACTORS:
            raise HTTPException(status_code=400, detail=f"Unknown shape: {shape}")
        
        # Calculate LSPR properties
        lspr_peak = calculate_lspr_peak(material, size, shape)
        ef_exponent = mie_enhancement_factor(size, material, excitation)
        fwhm = calculate_fwhm(material, size)
        
        # Apply shape enhancement factor
        ef_exponent += np.log10(SHAPE_FACTORS[shape]['ef_factor'])
        ef_magnitude = f"~10^{ef_exponent:.1f}"
        
        # Generate spectrum data (Lorentzian peak)
        wavelengths = np.linspace(300, 900, 500)
        gamma = fwhm / 2
        spectrum = gamma**2 / ((wavelengths - lspr_peak)**2 + gamma**2)
        spectrum = spectrum / np.max(spectrum)  # Normalize
        
        # Generate recommendation
        wavelength_match = abs(excitation - lspr_peak)
        if wavelength_match < 100:
            recommendation = (
                f"Excellent configuration! {MATERIAL_PARAMS[material]['name']} {shape} "
                f"nanoparticles show strong coupling at {excitation} nm excitation. "
                f"Expected enhancement: {ef_magnitude}"
            )
        elif wavelength_match < 200:
            recommendation = (
                f"Good enhancement expected with {MATERIAL_PARAMS[material]['name']} {shape}. "
                f"Consider adjusting particle size to shift LSPR peak closer to {excitation} nm. "
                f"Current LSPR at {lspr_peak:.0f} nm."
            )
        else:
            alt_material = 'Au' if material == 'Ag' else 'Ag'
            recommendation = (
                f"Wavelength mismatch detected. LSPR peak ({lspr_peak:.0f} nm) is far from "
                f"excitation ({excitation} nm). Recommendations:\n"
                f"1. Use {MATERIAL_PARAMS[alt_material]['name']} nanoparticles\n"
                f"2. Adjust excitation to {lspr_peak:.0f} nm\n"
                f"3. Use nanostars for broader spectral response"
            )
        
        return SimulateResponse(
            success=True,
            lspr_peak=lspr_peak,
            enhancement_factor=ef_magnitude,
            fwhm=fwhm,
            wavelengths=wavelengths.tolist(),
            spectrum=spectrum.tolist(),
            recommendation=recommendation,
            material_properties=MaterialProperties(
                name=MATERIAL_PARAMS[material]['name'],
                epsilon_inf=MATERIAL_PARAMS[material]['epsilon_inf'],
                omega_p=MATERIAL_PARAMS[material]['omega_p'],
                gamma=MATERIAL_PARAMS[material]['gamma'],
            ),
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}")


@router.get("/simulate/materials")
async def get_materials():
    """Get available materials and their properties"""
    return {
        "materials": [
            {
                "id": key,
                "name": val['name'],
                "lspr_peak_base": val['lspr_peak_base'],
                "properties": {
                    "epsilon_inf": val['epsilon_inf'],
                    "omega_p": val['omega_p'],
                    "gamma": val['gamma'],
                }
            }
            for key, val in MATERIAL_PARAMS.items()
        ],
        "shapes": [
            {"id": key, "peak_shift": val['peak_shift'], "ef_factor": val['ef_factor']}
            for key, val in SHAPE_FACTORS.items()
        ],
    }
