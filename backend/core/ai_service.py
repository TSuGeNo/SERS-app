"""
AI Service - OpenRouter integration for multi-model SERS analysis
Supports OpenAI, Anthropic, and Google models via OpenRouter
Includes streaming support, model verification, and fallback logic
"""

import os
import json
import httpx
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timedelta

from core.config import settings


class ModelStatus(str, Enum):
    """Model availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NOT_VERIFIED = "not_verified"


class AIModel:
    """AI Model configuration with status tracking"""
    
    # Using models that work on OpenRouter free/low-cost tier
    MODELS = {
        "google/gemini-2.0-flash-exp:free": {
            "name": "Gemini 2.0 Flash",
            "provider": "Google",
            "fallback_order": 1,  # Free model - highest priority
        },
        "meta-llama/llama-3.3-70b-instruct:free": {
            "name": "Llama 3.3 70B",
            "provider": "Meta",
            "fallback_order": 2,  # Free model
        },
        "qwen/qwen-2.5-72b-instruct:free": {
            "name": "Qwen 2.5 72B",
            "provider": "Alibaba",
            "fallback_order": 3,  # Free model
        },
    }
    
    @classmethod
    def get_fallback_order(cls) -> List[str]:
        """Get models in fallback priority order"""
        return sorted(
            cls.MODELS.keys(),
            key=lambda x: cls.MODELS[x]["fallback_order"]
        )


# Comprehensive SERS Analysis System Prompt
SERS_SYSTEM_PROMPT = """You are **SERS-Insight AI**, an expert assistant specializing in Surface-Enhanced Raman Spectroscopy (SERS) analysis. You help researchers and scientists analyze spectral data, identify molecules, understand SERS phenomena, and develop analytical methods.

## Your Core Expertise:

### 1. Peak Detection & Spectral Analysis
- Identifying characteristic Raman peaks and their molecular origins
- Peak fitting (Lorentzian, Gaussian, Voigt profiles)
- Band assignment based on vibrational modes
- Resolution enhancement techniques

### 2. Molecule Identification & Database Matching
- Matching spectral signatures to known compounds
- Confidence scoring and peak overlap analysis
- Multi-component mixture analysis
- Unknown compound characterization

### 3. LSPR & Enhancement Mechanisms
- Electromagnetic (EM) enhancement theory
- Chemical enhancement (CT) mechanisms
- Hot spot identification and optimization
- Substrate design principles (nanoparticles, nanostructures)

### 4. Data Preprocessing Pipeline
- Baseline correction (polynomial, ALS, SNIP, ModPoly)
- Smoothing (Savitzky-Golay, Gaussian, moving average)
- Normalization (min-max, SNV, MSC)
- Cosmic ray removal
- Spectral calibration

### 5. Machine Learning for SERS
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Classification (SVM, Random Forest, KNN, Neural Networks)
- Clustering (K-means, DBSCAN, hierarchical)
- Deep learning (1D-CNN, autoencoders, transformers)
- Transfer learning for small datasets

### 6. Quantitative Analysis
- Calibration curve development
- Limit of detection (LOD) and quantification (LOQ)
- Internal standard methods
- Multivariate calibration (PLS, PCR)

## Comprehensive SERS Peak Reference Library:

### Common SERS Probe Molecules:
| Molecule | Key Peaks (cm⁻¹) | Primary Assignments |
|----------|------------------|---------------------|
| **R6G (Rhodamine 6G)** | 611, 773, 1127, 1185, 1311, 1363, 1509, 1575, 1649 | Xanthene ring modes, C-C stretches |
| **Crystal Violet** | 441, 525, 724, 795, 915, 1175, 1371, 1587, 1619 | Phenyl ring modes |
| **Nile Blue** | 495, 546, 592, 663, 1074, 1159, 1430, 1492, 1641 | Benzophenoxazine modes |
| **Methylene Blue** | 449, 501, 596, 770, 1040, 1155, 1302, 1394, 1623 | Phenothiazine modes |
| **4-MBA (4-Mercaptobenzoic acid)** | 525, 710, 840, 1015, 1078, 1180, 1485, 1587 | Benzene ring, COO- modes |
| **4-ATP (4-Aminothiophenol)** | 390, 635, 1080, 1180, 1435, 1490, 1590 | Benzene ring, NH2 modes |

### Biomolecule Raman Markers:
| Biomolecule | Key Peaks (cm⁻¹) | Assignment |
|-------------|------------------|------------|
| **DNA/RNA** | 670-680, 720-730, 780-790, 1090, 1240-1260, 1320-1340, 1480, 1575 | Nucleobases, phosphate backbone |
| **Proteins (Amide)** | 1665 (Amide I), 1555 (Amide II), 1235 (Amide III) | Peptide bond vibrations |
| **Phenylalanine** | 1003 | Symmetric ring breathing |
| **Tyrosine** | 830, 850 (Fermi doublet), 1615 | Ring modes, OH |
| **Tryptophan** | 760, 880, 1012, 1340, 1360, 1555 | Indole ring modes |
| **Lipids** | 1065, 1130, 1300, 1440, 1655, 2850, 2880 | CH2/CH3 deformations, C=C |

## Response Guidelines:
1. **Scientific Rigor**: Always cite wavenumber values accurately (±5 cm⁻¹ tolerance for SERS)
2. **Step-by-Step Analysis**: Break down complex procedures into numbered steps
3. **Code Generation**: Provide executable Python code using scipy, numpy, sklearn, matplotlib
4. **Uncertainty Quantification**: Express confidence levels and note limitations
5. **Visual Guidance**: Recommend appropriate visualization types
6. **Literature Context**: Reference established SERS methodology when relevant
7. **Practical Tips**: Include experimental considerations and troubleshooting

Remember: SERS enhancement can cause peak shifts (typically 5-20 cm⁻¹) and intensity changes compared to normal Raman. Always consider substrate-molecule interactions."""


# Specialized prompts for specific analysis types
ANALYSIS_PROMPTS = {
    "peak_detection": """Focus on: 1) Identifying all significant peaks above noise level, 2) Estimating peak positions, heights, and widths, 3) Providing peak assignments based on SERS reference library. Include a comprehensive table and Python code.""",
    "molecule_identification": """Focus on: 1) Matching detected peaks against known reference spectra, 2) Calculating confidence scores, 3) Considering SERS-induced peak shifts. Provide a ranked list of candidate molecules.""",
    "baseline_correction": """Focus on: 1) Identifying baseline artifact type, 2) Recommending appropriate algorithm (ALS, polynomial, SNIP), 3) Providing parameter tuning guidance. Include Python code.""",
    "machine_learning": """Focus on: 1) Data preprocessing, 2) Algorithm selection, 3) Cross-validation strategy, 4) Model interpretation. Provide complete sklearn code with best practices.""",
}


def format_data_context(data_info: Optional[Dict[str, Any]] = None) -> str:
    """Format data context for the AI prompt"""
    if not data_info:
        return ""
    
    context = "\n\n## 📊 Current Data Context:\n"
    
    if data_info.get("filename"):
        context += f"- **File**: `{data_info['filename']}`\n"
    
    if data_info.get("data_type"):
        context += f"- **Data Type**: {data_info['data_type']}\n"
    
    if data_info.get("data_points"):
        context += f"- **Data Points**: {data_info['data_points']:,}\n"
    
    if data_info.get("wavenumber_range"):
        wn_range = data_info["wavenumber_range"]
        context += f"- **Spectral Range**: {wn_range[0]:.1f} - {wn_range[1]:.1f} cm⁻¹\n"
    
    if data_info.get("detected_peaks"):
        peaks = data_info["detected_peaks"]
        if isinstance(peaks, list) and len(peaks) > 0:
            peaks_str = ', '.join([f"{p}" for p in peaks[:12]])
            context += f"- **Auto-detected Peaks**: {peaks_str} cm⁻¹\n"
    
    return context


def detect_analysis_type(message: str) -> Optional[str]:
    """Detect the type of analysis requested from the message"""
    lower_msg = message.lower()
    
    if any(kw in lower_msg for kw in ['peak', 'find peak', 'detect peak']):
        return "peak_detection"
    elif any(kw in lower_msg for kw in ['identify', 'what molecule', 'which compound']):
        return "molecule_identification"
    elif any(kw in lower_msg for kw in ['baseline', 'background', 'fluorescence']):
        return "baseline_correction"
    elif any(kw in lower_msg for kw in ['ml', 'machine learning', 'classify', 'cluster']):
        return "machine_learning"
    
    return None


class AIService:
    """OpenRouter-based AI service for SERS analysis with verification and fallback"""
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    VERIFICATION_CACHE_DURATION = timedelta(minutes=5)
    
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY if hasattr(settings, 'OPENROUTER_API_KEY') else ""
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
        # Model status tracking
        self._model_status: Dict[str, Dict[str, Any]] = {}
        self._last_verification: Optional[datetime] = None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their status"""
        models = []
        
        for model_id, info in AIModel.MODELS.items():
            status = self._model_status.get(model_id, {})
            models.append({
                "id": model_id,
                "name": info["name"],
                "provider": info["provider"],
                "status": status.get("status", ModelStatus.NOT_VERIFIED),
                "last_error": status.get("error"),
                "last_verified": status.get("verified_at"),
            })
        
        if not self.api_key:
            models.append({
                "id": "demo",
                "name": "Demo Mode (No API Key)",
                "provider": "Local",
                "status": ModelStatus.AVAILABLE,
            })
        
        return models
    
    async def verify_api_key(self) -> Tuple[bool, str]:
        """Verify if the OpenRouter API key is valid"""
        if not self.api_key:
            return False, "No API key configured. Using demo mode."
        
        try:
            # Test with a minimal request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            response = await self.http_client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
            )
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"API error: {response.status_code}"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    async def verify_model(self, model_id: str) -> Dict[str, Any]:
        """Verify if a specific model is reachable and working"""
        result = {
            "model": model_id,
            "status": ModelStatus.NOT_VERIFIED,
            "verified_at": datetime.now().isoformat(),
            "response_time_ms": None,
            "error": None,
        }
        
        if not self.api_key:
            result["status"] = ModelStatus.UNAVAILABLE
            result["error"] = "No API key configured"
            return result
        
        try:
            start_time = datetime.now()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sers-insight.local",
                "X-Title": "SERS-Insight Platform",
            }
            
            # Send minimal test message
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Test. Reply with OK."}],
                "max_tokens": 10,
                "temperature": 0,
            }
            
            response = await self.http_client.post(
                self.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            result["response_time_ms"] = round(response_time, 2)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    result["status"] = ModelStatus.AVAILABLE
                else:
                    result["status"] = ModelStatus.ERROR
                    result["error"] = "Empty response from model"
            else:
                result["status"] = ModelStatus.UNAVAILABLE
                try:
                    error_data = response.json()
                    result["error"] = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                except:
                    result["error"] = f"HTTP {response.status_code}"
        
        except httpx.TimeoutException:
            result["status"] = ModelStatus.UNAVAILABLE
            result["error"] = "Request timeout"
        except Exception as e:
            result["status"] = ModelStatus.ERROR
            result["error"] = str(e)
        
        # Cache the result
        self._model_status[model_id] = result
        
        return result
    
    async def verify_all_models(self) -> Dict[str, Any]:
        """Verify all configured models"""
        results = {
            "api_key_valid": False,
            "api_key_message": "",
            "models": {},
            "verified_at": datetime.now().isoformat(),
        }
        
        # First verify API key
        key_valid, key_message = await self.verify_api_key()
        results["api_key_valid"] = key_valid
        results["api_key_message"] = key_message
        
        if not key_valid:
            for model_id in AIModel.MODELS:
                results["models"][model_id] = {
                    "status": ModelStatus.UNAVAILABLE,
                    "error": "API key invalid",
                }
            return results
        
        # Verify each model
        for model_id in AIModel.MODELS:
            results["models"][model_id] = await self.verify_model(model_id)
        
        self._last_verification = datetime.now()
        
        return results
    
    def _get_best_available_model(self, preferred_model: str) -> Tuple[str, Optional[str]]:
        """Get the best available model, with fallback if needed"""
        # If preferred model is available, use it
        preferred_status = self._model_status.get(preferred_model, {})
        if preferred_status.get("status") == ModelStatus.AVAILABLE:
            return preferred_model, None
        
        # Try fallback models in order
        for model_id in AIModel.get_fallback_order():
            if model_id == preferred_model:
                continue
            status = self._model_status.get(model_id, {})
            if status.get("status") == ModelStatus.AVAILABLE:
                fallback_info = AIModel.MODELS[model_id]
                return model_id, f"Switched to {fallback_info['name']} (fallback)"
        
        # If no verification done, just try the preferred model
        if not self._model_status:
            return preferred_model, None
        
        return preferred_model, "Warning: Model availability not verified"
    
    def _build_messages(
        self,
        message: str,
        data_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the message list for the API call"""
        analysis_type = detect_analysis_type(message)
        
        system_prompt = SERS_SYSTEM_PROMPT
        if analysis_type and analysis_type in ANALYSIS_PROMPTS:
            system_prompt += f"\n\n## Current Task:\n{ANALYSIS_PROMPTS[analysis_type]}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-10:])
        
        data_context_str = format_data_context(data_context)
        full_message = f"{message}{data_context_str}" if data_context_str else message
        
        messages.append({"role": "user", "content": full_message})
        
        return messages
    
    async def generate_response(
        self,
        message: str,
        model: str = "anthropic/claude-sonnet-4.5",
        data_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using OpenRouter API with fallback support.
        
        Returns:
            Tuple of (response_text, metadata)
            metadata includes: model_used, fallback_message, error if any
        """
        metadata = {
            "model_requested": model,
            "model_used": model,
            "fallback_message": None,
            "error": None,
        }
        
        if not self.api_key:
            response = self._generate_demo_response(message, data_context)
            metadata["model_used"] = "demo"
            metadata["fallback_message"] = "Using demo mode (no API key configured)"
            return response, metadata
        
        # Get best available model with fallback
        actual_model, fallback_msg = self._get_best_available_model(model)
        metadata["model_used"] = actual_model
        metadata["fallback_message"] = fallback_msg
        
        # Try the request
        max_retries = 2
        models_tried = [actual_model]
        
        for attempt in range(max_retries + 1):
            try:
                messages = self._build_messages(message, data_context, conversation_history)
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://sers-insight.local",
                    "X-Title": "SERS-Insight Platform",
                }
                
                payload = {
                    "model": actual_model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                }
                
                response = await self.http_client.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Update model status to available
                    self._model_status[actual_model] = {
                        "status": ModelStatus.AVAILABLE,
                        "verified_at": datetime.now().isoformat(),
                    }
                    
                    return content, metadata
                
                else:
                    # Model failed, mark as unavailable
                    self._model_status[actual_model] = {
                        "status": ModelStatus.UNAVAILABLE,
                        "error": f"HTTP {response.status_code}",
                        "verified_at": datetime.now().isoformat(),
                    }
                    
                    # Try next fallback model
                    for fallback_model in AIModel.get_fallback_order():
                        if fallback_model not in models_tried:
                            actual_model = fallback_model
                            models_tried.append(fallback_model)
                            fallback_info = AIModel.MODELS[fallback_model]
                            metadata["model_used"] = fallback_model
                            metadata["fallback_message"] = f"Switched to {fallback_info['name']} after error"
                            break
                    else:
                        # No more models to try
                        break
            
            except Exception as e:
                metadata["error"] = str(e)
                
                # Mark current model as error
                self._model_status[actual_model] = {
                    "status": ModelStatus.ERROR,
                    "error": str(e),
                    "verified_at": datetime.now().isoformat(),
                }
                
                # Try next fallback
                for fallback_model in AIModel.get_fallback_order():
                    if fallback_model not in models_tried:
                        actual_model = fallback_model
                        models_tried.append(fallback_model)
                        fallback_info = AIModel.MODELS[fallback_model]
                        metadata["model_used"] = fallback_model
                        metadata["fallback_message"] = f"Switched to {fallback_info['name']} after error"
                        break
                else:
                    break
        
        # All models failed, use demo response
        response = self._generate_demo_response(message, data_context)
        metadata["model_used"] = "demo"
        metadata["fallback_message"] = f"All models failed. Using demo mode. Tried: {', '.join(models_tried)}"
        return response, metadata
    
    async def generate_response_stream(
        self,
        message: str,
        model: str = "anthropic/claude-sonnet-4.5",
        data_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a streaming response with metadata events"""
        
        if not self.api_key:
            yield {"type": "fallback", "message": "Using demo mode (no API key)"}
            demo_response = self._generate_demo_response(message, data_context)
            for char in demo_response:
                yield {"type": "chunk", "content": char}
            yield {"type": "end", "model_used": "demo"}
            return
        
        # Get best available model
        actual_model, fallback_msg = self._get_best_available_model(model)
        
        if fallback_msg:
            yield {"type": "fallback", "message": fallback_msg}
        
        yield {"type": "start", "model_used": actual_model}
        
        try:
            messages = self._build_messages(message, data_context, conversation_history)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sers-insight.local",
                "X-Title": "SERS-Insight Platform",
            }
            
            payload = {
                "model": actual_model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.7,
                "stream": True,
            }
            
            async with self.http_client.stream(
                "POST",
                self.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    yield {"type": "error", "message": f"API error: {response.status_code}"}
                    # Fall back to demo
                    yield {"type": "fallback", "message": "Switching to demo mode"}
                    demo_response = self._generate_demo_response(message, data_context)
                    for char in demo_response:
                        yield {"type": "chunk", "content": char}
                    yield {"type": "end", "model_used": "demo"}
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield {"type": "chunk", "content": content}
                        except json.JSONDecodeError:
                            continue
            
            yield {"type": "end", "model_used": actual_model}
        
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            yield {"type": "fallback", "message": "Switching to demo mode due to error"}
            demo_response = self._generate_demo_response(message, data_context)
            for char in demo_response:
                yield {"type": "chunk", "content": char}
            yield {"type": "end", "model_used": "demo"}
    
    def _generate_demo_response(
        self,
        message: str,
        data_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a demo response when no API key is configured"""
        lower_msg = message.lower()
        
        if data_context and data_context.get("filename"):
            filename = data_context["filename"]
            data_points = data_context.get("data_points", "unknown")
            
            if "peak" in lower_msg or "detect" in lower_msg:
                return f"""## 🔬 Peak Detection Analysis

**Analyzing**: `{filename}`
**Data Points**: {data_points}

### Detected Peaks:

| Peak # | Position (cm⁻¹) | Intensity | Assignment |
|--------|-----------------|-----------|------------|
| 1 | 611 ± 2 | High | C-C-C ring bend (R6G) |
| 2 | 773 ± 3 | Medium | C-H out-of-plane |
| 3 | 1363 ± 2 | High | Aromatic C-C stretch |
| 4 | 1509 ± 2 | Very High | Aromatic C-C stretch |
| 5 | 1649 ± 2 | Medium | C=C stretch |

### Python Implementation:

```python
from scipy.signal import find_peaks, savgol_filter
import numpy as np

def detect_sers_peaks(wavenumber, intensity):
    # Smooth the spectrum
    smoothed = savgol_filter(intensity, 11, 3)
    
    # Find peaks
    peaks, props = find_peaks(
        smoothed,
        prominence=0.1 * np.max(smoothed),
        distance=10
    )
    
    return wavenumber[peaks], smoothed[peaks]

# Usage
peak_positions, peak_intensities = detect_sers_peaks(wn, inten)
print(f"Found {{len(peak_positions)}} peaks")
```

### Analysis Summary:
**Molecule Identified**: Rhodamine 6G (R6G) with 95% confidence"""

            else:
                return f"""## 📊 Data Ready for Analysis

**File**: `{filename}`
**Data Points**: {data_points}

Your SERS spectrum is loaded. Try:
- "Detect peaks in my spectrum"
- "What molecule is this?"
- "Apply baseline correction"
- "Generate ML classification code" """

        # Default welcome
        return """## 🔬 Welcome to SERS-Insight AI!

I'm your expert assistant for Surface-Enhanced Raman Spectroscopy analysis.

### What I Can Do:
- **Peak Detection** - Find and assign spectral peaks
- **Molecule ID** - Match spectra to known compounds
- **ML Analysis** - Classification, clustering, PCA
- **Code Generation** - Python scripts for your analysis

### Get Started:
1. Upload your SERS spectrum data
2. Ask questions about your data
3. Get code, visualizations, and insights

**Upload data or ask about SERS!**"""
    
    async def close(self):
        """Close the HTTP client"""
        await self.http_client.aclose()


# Global AI service instance
ai_service = AIService()
