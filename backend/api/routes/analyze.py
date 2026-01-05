"""Analysis framework endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    FrameworkInfo,
    MoleculeDetectionResult,
    ClassificationResult,
)

router = APIRouter()

# Reference peak library for common SERS probes
PEAK_LIBRARY = {
    'R6G': {
        'name': 'Rhodamine 6G',
        'peaks': [611, 773, 1127, 1185, 1311, 1363, 1509, 1575, 1649],
        'tolerance': 15,  # cm^-1
    },
    'CV': {
        'name': 'Crystal Violet',
        'peaks': [441, 525, 724, 795, 915, 1175, 1371, 1587, 1619],
        'tolerance': 15,
    },
    'NB': {
        'name': 'Nile Blue',
        'peaks': [495, 546, 592, 663, 1074, 1159, 1430, 1492, 1641],
        'tolerance': 15,
    },
    'MB': {
        'name': 'Methylene Blue',
        'peaks': [449, 501, 596, 770, 1040, 1155, 1302, 1394, 1623],
        'tolerance': 15,
    },
}


def detect_molecule(wavenumber: np.ndarray, intensity: np.ndarray, target: str = 'auto') -> MoleculeDetectionResult:
    """
    Detect molecule based on characteristic SERS peaks.
    
    Parameters:
        wavenumber: Wavenumber values
        intensity: Intensity values
        target: Target molecule or 'auto' for auto-detection
    
    Returns:
        Detection result with matched peaks and confidence
    """
    from scipy.signal import find_peaks
    
    # Find peaks in the spectrum
    peaks_idx, _ = find_peaks(intensity, prominence=0.1 * np.max(intensity), distance=10)
    detected_peaks = wavenumber[peaks_idx]
    
    best_match = None
    best_score = 0
    all_matches = []
    
    molecules_to_check = [target] if target != 'auto' else list(PEAK_LIBRARY.keys())
    
    for mol_id in molecules_to_check:
        if mol_id not in PEAK_LIBRARY:
            continue
            
        mol_info = PEAK_LIBRARY[mol_id]
        ref_peaks = mol_info['peaks']
        tolerance = mol_info['tolerance']
        
        matched_peaks = []
        for ref_peak in ref_peaks:
            for det_peak in detected_peaks:
                if abs(det_peak - ref_peak) <= tolerance:
                    matched_peaks.append({
                        'reference': ref_peak,
                        'detected': float(det_peak),
                        'offset': float(det_peak - ref_peak),
                    })
                    break
        
        match_ratio = len(matched_peaks) / len(ref_peaks)
        confidence = match_ratio * 100
        
        all_matches.append({
            'molecule_id': mol_id,
            'molecule_name': mol_info['name'],
            'matched_peaks': matched_peaks,
            'total_reference_peaks': len(ref_peaks),
            'confidence': confidence,
        })
        
        if confidence > best_score:
            best_score = confidence
            best_match = {
                'molecule_id': mol_id,
                'molecule_name': mol_info['name'],
                'confidence': confidence,
                'matched_peaks': matched_peaks,
            }
    
    return MoleculeDetectionResult(
        detected=best_match is not None and best_score > 50,
        best_match=best_match,
        all_matches=sorted(all_matches, key=lambda x: x['confidence'], reverse=True),
    )


def run_pca_classification(
    spectra: np.ndarray,
    labels: List[str],
    n_components: int = 5,
    classifier: str = 'svm',
    cross_validate: bool = True,
) -> ClassificationResult:
    """
    Run PCA + classification pipeline.
    
    Parameters:
        spectra: 2D array of spectra (samples x features)
        labels: Class labels
        n_components: Number of PCA components
        classifier: 'svm' or 'rf'
        cross_validate: Whether to run cross-validation
    
    Returns:
        Classification result with metrics
    """
    # Standardize features
    scaler = StandardScaler()
    spectra_scaled = scaler.fit_transform(spectra)
    
    # PCA
    pca = PCA(n_components=n_components)
    spectra_pca = pca.fit_transform(spectra_scaled)
    
    # Classifier
    if classifier == 'svm':
        clf = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-validation
    metrics = {}
    if cross_validate:
        cv_scores = cross_val_score(clf, spectra_pca, labels, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = float(np.mean(cv_scores))
        metrics['cv_accuracy_std'] = float(np.std(cv_scores))
    
    # Fit final model
    clf.fit(spectra_pca, labels)
    
    # Get predictions and probabilities
    predictions = clf.predict(spectra_pca)
    
    # Metrics
    unique_labels = list(set(labels))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    
    return ClassificationResult(
        success=True,
        classifier=classifier,
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        metrics=metrics,
        confusion_matrix=cm.tolist(),
        class_labels=unique_labels,
        pca_components=spectra_pca.tolist(),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_data(request: AnalyzeRequest):
    """
    Run analysis framework on input data.
    
    Supported frameworks:
    - molecule_detection: Detect known SERS molecules
    - biomolecule_classification: PCA + SVM/RF classification
    - spectral_unmixing: NMF-based component separation
    """
    try:
        framework = request.framework
        
        if framework == 'molecule_detection':
            wavenumber = np.array(request.wavenumber)
            intensity = np.array(request.intensity)
            target = request.parameters.get('target_molecule', 'auto')
            
            result = detect_molecule(wavenumber, intensity, target)
            
            return AnalyzeResponse(
                success=True,
                framework=framework,
                result=result.dict(),
            )
        
        elif framework == 'biomolecule_classification':
            spectra = np.array(request.spectra)
            labels = request.labels
            n_components = request.parameters.get('n_components', 5)
            classifier = request.parameters.get('classifier', 'svm')
            
            result = run_pca_classification(
                spectra, labels, n_components, classifier
            )
            
            return AnalyzeResponse(
                success=True,
                framework=framework,
                result=result.dict(),
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown framework: {framework}")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")


@router.get("/frameworks")
async def list_frameworks():
    """List available analysis frameworks"""
    frameworks = [
        FrameworkInfo(
            id='molecule_detection',
            name='Molecule Detection',
            description='Detect known SERS molecules (R6G, Crystal Violet, etc.) by peak matching',
            category='detection',
            parameters=[
                {'name': 'target_molecule', 'type': 'select', 'options': ['auto'] + list(PEAK_LIBRARY.keys())},
            ],
        ),
        FrameworkInfo(
            id='biomolecule_classification',
            name='Biomolecule Classification',
            description='PCA + classifier for biomolecule identification',
            category='classification',
            parameters=[
                {'name': 'n_components', 'type': 'range', 'min': 2, 'max': 20, 'default': 5},
                {'name': 'classifier', 'type': 'select', 'options': ['svm', 'rf']},
            ],
        ),
        FrameworkInfo(
            id='pathogen_detection',
            name='Pathogen Detection',
            description='CNN-based bacterial species classification',
            category='classification',
            parameters=[
                {'name': 'model', 'type': 'select', 'options': ['cnn1d', 'ensemble']},
                {'name': 'augmentation', 'type': 'boolean', 'default': True},
            ],
        ),
        FrameworkInfo(
            id='spectral_unmixing',
            name='Spectral Unmixing',
            description='NMF-based separation of mixed spectra into pure components',
            category='unmixing',
            parameters=[
                {'name': 'n_components', 'type': 'range', 'min': 2, 'max': 10, 'default': 3},
                {'name': 'method', 'type': 'select', 'options': ['nmf', 'ica']},
            ],
        ),
    ]
    
    return {"frameworks": [f.dict() for f in frameworks]}
