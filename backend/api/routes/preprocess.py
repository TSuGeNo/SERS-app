"""Data preprocessing endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.signal import savgol_filter, find_peaks

from schemas.preprocess import (
    PreprocessRequest,
    PreprocessResponse,
    PreprocessingOptions,
    PeakInfo,
)

router = APIRouter()


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    
    Parameters:
        y: Input signal
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (smaller = more asymmetric)
        niter: Number of iterations
    
    Returns:
        Estimated baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = splu(Z.tocsc()).solve(w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z


def normalize_spectrum(y: np.ndarray, method: str = 'vector') -> np.ndarray:
    """
    Normalize spectrum intensity.
    
    Parameters:
        y: Input signal
        method: 'vector', 'max', or 'minmax'
    
    Returns:
        Normalized signal
    """
    if method == 'vector':
        norm = np.linalg.norm(y)
        return y / norm if norm > 0 else y
    elif method == 'max':
        max_val = np.max(y)
        return y / max_val if max_val > 0 else y
    elif method == 'minmax':
        min_val, max_val = np.min(y), np.max(y)
        if max_val - min_val > 0:
            return (y - min_val) / (max_val - min_val)
        return y
    else:
        return y


def detect_peaks_in_spectrum(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    height: Optional[float] = None,
    prominence: float = 0.1,
    distance: int = 10,
) -> list[PeakInfo]:
    """
    Detect peaks in a spectrum.
    
    Parameters:
        wavenumber: Wavenumber values
        intensity: Intensity values
        height: Minimum peak height
        prominence: Minimum peak prominence
        distance: Minimum distance between peaks
    
    Returns:
        List of detected peaks
    """
    # Find peaks
    peaks, properties = find_peaks(
        intensity,
        height=height,
        prominence=prominence * np.max(intensity),
        distance=distance,
    )
    
    peak_list = []
    for i, peak_idx in enumerate(peaks):
        peak_list.append(PeakInfo(
            wavenumber=float(wavenumber[peak_idx]),
            intensity=float(intensity[peak_idx]),
            prominence=float(properties['prominences'][i]) if 'prominences' in properties else 0,
            width=float(properties.get('widths', [0])[i] if i < len(properties.get('widths', [])) else 0),
        ))
    
    return peak_list


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(request: PreprocessRequest):
    """
    Apply preprocessing pipeline to spectrum data.
    
    Steps applied in order:
    1. Baseline correction (Asymmetric Least Squares)
    2. Smoothing (Savitzky-Golay filter)
    3. Normalization (vector, max, or minmax)
    4. Peak detection (optional)
    """
    try:
        # Convert input data to numpy arrays
        wavenumber = np.array(request.wavenumber)
        intensity = np.array(request.intensity)
        
        original_intensity = intensity.copy()
        options = request.options or PreprocessingOptions()
        
        # Baseline correction
        if options.baseline_correction:
            baseline = baseline_als(
                intensity,
                lam=options.baseline_lambda,
                p=options.baseline_p,
            )
            intensity = intensity - baseline
        
        # Smoothing
        if options.smoothing:
            intensity = savgol_filter(
                intensity,
                window_length=options.smoothing_window,
                polyorder=options.smoothing_polyorder,
            )
        
        # Normalization
        if options.normalization != 'none':
            intensity = normalize_spectrum(intensity, options.normalization)
        
        # Peak detection
        peaks = []
        if options.peak_detection:
            peaks = detect_peaks_in_spectrum(
                wavenumber,
                intensity,
                prominence=options.peak_prominence,
                distance=options.peak_distance,
            )
        
        return PreprocessResponse(
            success=True,
            wavenumber=wavenumber.tolist(),
            original_intensity=original_intensity.tolist(),
            processed_intensity=intensity.tolist(),
            peaks=peaks,
            applied_steps=[
                step for step, applied in [
                    ('baseline_correction', options.baseline_correction),
                    ('smoothing', options.smoothing),
                    ('normalization', options.normalization != 'none'),
                    ('peak_detection', options.peak_detection),
                ] if applied
            ],
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")


@router.get("/preprocess/options")
async def get_preprocessing_options():
    """Get available preprocessing options and their defaults"""
    return {
        "baseline_correction": {
            "description": "Asymmetric Least Squares baseline correction",
            "parameters": {
                "lambda": {"default": 1e5, "range": [1e3, 1e9], "description": "Smoothness parameter"},
                "p": {"default": 0.01, "range": [0.001, 0.1], "description": "Asymmetry parameter"},
            }
        },
        "smoothing": {
            "description": "Savitzky-Golay smoothing filter",
            "parameters": {
                "window": {"default": 11, "range": [5, 51], "description": "Window length (odd number)"},
                "polyorder": {"default": 3, "range": [1, 5], "description": "Polynomial order"},
            }
        },
        "normalization": {
            "description": "Intensity normalization",
            "options": ["none", "vector", "max", "minmax"],
            "default": "vector",
        },
        "peak_detection": {
            "description": "Automatic peak detection",
            "parameters": {
                "prominence": {"default": 0.1, "range": [0.01, 0.5], "description": "Minimum prominence (fraction of max)"},
                "distance": {"default": 10, "range": [1, 50], "description": "Minimum distance between peaks"},
            }
        }
    }
