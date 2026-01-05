"""AI Chat endpoints with multi-model support, streaming, verification, and fallback"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import uuid
from datetime import datetime
import json
import numpy as np
from io import StringIO
import pandas as pd
import asyncio

from core.ai_service import ai_service, ModelStatus
from core.config import settings

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str
    session_id: Optional[str] = None
    model: Optional[str] = "anthropic/claude-sonnet-4.5"
    data_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    response: str
    model_used: str
    model_requested: str
    fallback_message: Optional[str] = None


class StreamChatRequest(BaseModel):
    """Request for streaming chat"""
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "anthropic/claude-sonnet-4.5"
    data_context: Optional[Dict[str, Any]] = None


class DataAnalysisRequest(BaseModel):
    """Request for analyzing uploaded data"""
    wavenumber: Optional[List[float]] = None
    intensity: Optional[List[float]] = None
    filename: Optional[str] = None
    raw_data: Optional[str] = None


class VerifyModelRequest(BaseModel):
    """Request to verify a specific model"""
    model: str


# In-memory chat sessions
CHAT_SESSIONS: Dict[str, List[Dict]] = {}


def analyze_spectrum_data(wavenumber: List[float], intensity: List[float]) -> Dict[str, Any]:
    """Analyze spectrum and extract useful features"""
    from scipy.signal import find_peaks, savgol_filter
    
    wn = np.array(wavenumber)
    inten = np.array(intensity)
    
    if len(inten) > 11:
        smoothed = savgol_filter(inten, window_length=11, polyorder=3)
    else:
        smoothed = inten
    
    peaks_idx, properties = find_peaks(
        smoothed, 
        prominence=0.05 * np.max(smoothed),
        distance=5
    )
    
    detected_peaks = wn[peaks_idx].tolist() if len(peaks_idx) > 0 else []
    
    if len(inten) > 50:
        noise_region = inten[:50]
        signal_region = smoothed
        snr = float(np.max(signal_region) / np.std(noise_region)) if np.std(noise_region) > 0 else 100.0
    else:
        snr = 10.0
    
    return {
        "data_points": len(wavenumber),
        "wavenumber_range": [float(min(wavenumber)), float(max(wavenumber))],
        "intensity_range": [float(min(intensity)), float(max(intensity))],
        "detected_peaks": sorted(detected_peaks)[:15],
        "snr": snr,
    }


@router.post("/chat", response_model=ChatResponse)
async def send_message(message: ChatMessage):
    """Send a message and get AI response with fallback support"""
    session_id = message.session_id or str(uuid.uuid4())
    model = message.model or settings.DEFAULT_AI_MODEL
    
    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = []
    
    history = CHAT_SESSIONS[session_id]
    
    history.append({
        "role": "user",
        "content": message.content,
        "timestamp": datetime.now().isoformat(),
    })
    
    # Generate AI response with fallback
    response_text, metadata = await ai_service.generate_response(
        message=message.content,
        model=model,
        data_context=message.data_context,
        conversation_history=[
            {"role": m["role"], "content": m["content"]} 
            for m in history[-10:]
        ],
    )
    
    history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().isoformat(),
        "model": metadata["model_used"],
    })
    
    CHAT_SESSIONS[session_id] = history[-50:]
    
    return ChatResponse(
        success=True,
        message="Response generated",
        session_id=session_id,
        response=response_text,
        model_used=metadata["model_used"],
        model_requested=model,
        fallback_message=metadata.get("fallback_message"),
    )


@router.post("/chat/stream")
async def stream_message(request: StreamChatRequest):
    """Send a message and get streaming AI response with fallback notifications"""
    session_id = request.session_id or str(uuid.uuid4())
    model = request.model or settings.DEFAULT_AI_MODEL
    
    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = []
    
    history = CHAT_SESSIONS[session_id]
    
    history.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat(),
    })
    
    async def event_generator():
        """Generate SSE events for streaming response"""
        full_response = ""
        model_used = model
        
        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'model': model})}\n\n"
            
            async for event in ai_service.generate_response_stream(
                message=request.message,
                model=model,
                data_context=request.data_context,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]} 
                    for m in history[-10:]
                ],
            ):
                if event["type"] == "fallback":
                    yield f"data: {json.dumps({'type': 'fallback', 'message': event['message']})}\n\n"
                elif event["type"] == "chunk":
                    full_response += event["content"]
                    yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"
                elif event["type"] == "end":
                    model_used = event.get("model_used", model)
                
                await asyncio.sleep(0.005)
            
            history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat(),
                "model": model_used,
            })
            
            CHAT_SESSIONS[session_id] = history[-50:]
            
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id, 'model_used': model_used})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/chat/verify")
async def verify_all_models():
    """Verify API key and all configured models"""
    results = await ai_service.verify_all_models()
    
    available_count = sum(
        1 for m in results["models"].values() 
        if m.get("status") == ModelStatus.AVAILABLE
    )
    
    return {
        "success": True,
        "api_key_valid": results["api_key_valid"],
        "api_key_message": results["api_key_message"],
        "models": results["models"],
        "available_models": available_count,
        "total_models": len(results["models"]),
        "verified_at": results["verified_at"],
    }


@router.post("/chat/verify-model")
async def verify_single_model(request: VerifyModelRequest):
    """Verify a specific model"""
    result = await ai_service.verify_model(request.model)
    
    return {
        "success": result["status"] == ModelStatus.AVAILABLE,
        "model": request.model,
        "status": result["status"],
        "response_time_ms": result.get("response_time_ms"),
        "error": result.get("error"),
        "verified_at": result["verified_at"],
    }


@router.get("/chat/models")
async def get_available_models():
    """Get list of available AI models with their status"""
    models = ai_service.get_available_models()
    return {
        "success": True,
        "models": models,
        "default_model": settings.DEFAULT_AI_MODEL,
    }


@router.post("/chat/analyze-data")
async def analyze_data_for_chat(request: DataAnalysisRequest):
    """Analyze uploaded data and return context for chat"""
    try:
        data_context = {
            "filename": request.filename,
            "data_type": "SERS Spectrum",
        }
        
        if request.wavenumber and request.intensity:
            analysis = analyze_spectrum_data(request.wavenumber, request.intensity)
            data_context.update(analysis)
        
        elif request.raw_data:
            try:
                df = pd.read_csv(StringIO(request.raw_data))
                if len(df.columns) >= 2:
                    wavenumber = df.iloc[:, 0].tolist()
                    intensity = df.iloc[:, 1].tolist()
                    analysis = analyze_spectrum_data(wavenumber, intensity)
                    data_context.update(analysis)
                    data_context["columns"] = df.columns.tolist()
            except Exception:
                data_context["parse_error"] = "Could not parse data automatically"
        
        return {
            "success": True,
            "data_context": data_context,
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    return {
        "session_id": session_id, 
        "messages": CHAT_SESSIONS.get(session_id, [])
    }


@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in CHAT_SESSIONS:
        del CHAT_SESSIONS[session_id]
    return {"success": True, "message": "History cleared"}


@router.post("/chat/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a file and analyze it for chat context"""
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        try:
            df = pd.read_csv(StringIO(text_content))
            
            data_context = {
                "filename": file.filename,
                "data_type": "CSV Data",
                "rows": len(df),
                "columns": df.columns.tolist(),
            }
            
            if len(df.columns) >= 2:
                first_col = df.columns[0].lower()
                if 'wave' in first_col or 'raman' in first_col or 'cm' in first_col:
                    data_context["data_type"] = "SERS Spectrum"
                    wavenumber = df.iloc[:, 0].tolist()
                    intensity = df.iloc[:, 1].tolist()
                    analysis = analyze_spectrum_data(wavenumber, intensity)
                    data_context.update(analysis)
            
            return {
                "success": True,
                "data_context": data_context,
                "preview": df.head(5).to_dict(),
            }
        
        except Exception:
            return {
                "success": True,
                "data_context": {
                    "filename": file.filename,
                    "data_type": "Text File",
                    "size_bytes": len(content),
                },
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
