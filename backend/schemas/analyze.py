"""Analysis schemas"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AnalyzeRequest(BaseModel):
    framework: str
    wavenumber: Optional[List[float]] = None
    intensity: Optional[List[float]] = None
    spectra: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    parameters: Dict[str, Any] = {}

class MoleculeDetectionResult(BaseModel):
    detected: bool
    best_match: Optional[Dict[str, Any]] = None
    all_matches: List[Dict[str, Any]] = []

class ClassificationResult(BaseModel):
    success: bool
    classifier: str
    n_components: int
    explained_variance_ratio: List[float]
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    pca_components: List[List[float]]

class FrameworkInfo(BaseModel):
    id: str
    name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]

class AnalyzeResponse(BaseModel):
    success: bool
    framework: str
    result: Dict[str, Any]
