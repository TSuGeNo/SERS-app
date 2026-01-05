"""File upload endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime

from core.config import settings
from schemas.upload import UploadResponse, DatasetInfo, DatasetMetadata

router = APIRouter()


def detect_file_type(filename: str) -> str:
    """Detect file type from extension"""
    ext = filename.lower().split('.')[-1]
    type_map = {
        'csv': 'csv',
        'txt': 'txt',
        'xlsx': 'xlsx',
        'xls': 'xlsx',
        'json': 'json',
        'png': 'image',
        'jpg': 'image',
        'jpeg': 'image',
    }
    return type_map.get(ext, 'unknown')


def parse_spectrum_file(file_path: str, file_type: str) -> tuple[pd.DataFrame, dict]:
    """Parse spectrum data file and extract metadata"""
    metadata = {}
    
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'txt':
        # Try different delimiters
        for sep in ['\t', ' ', ',', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) >= 2:
                    break
            except:
                continue
        else:
            raise ValueError("Could not parse TXT file")
    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Extract metadata
    metadata['columns'] = list(df.columns)
    metadata['rows'] = len(df)
    metadata['has_headers'] = not df.columns[0].replace('.', '').replace('-', '').isdigit()
    
    # Detect wavenumber column
    wavenumber_cols = ['wavenumber', 'raman_shift', 'cm-1', 'cm^-1', 'x']
    for col in df.columns:
        if any(wc in col.lower() for wc in wavenumber_cols):
            metadata['wavenumber_column'] = col
            break
    else:
        # Assume first column is wavenumber
        metadata['wavenumber_column'] = df.columns[0]
    
    # Calculate wavenumber range
    wn_col = metadata['wavenumber_column']
    if wn_col in df.columns:
        metadata['wavenumber_range'] = [float(df[wn_col].min()), float(df[wn_col].max())]
    
    return df, metadata


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file for analysis.
    
    Supported formats:
    - CSV (comma-separated values)
    - TXT (tab or space delimited)
    - XLSX (Excel spreadsheet)
    - JSON (structured data)
    - Images (PNG, JPG for visual analysis)
    """
    # Validate file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_type = detect_file_type(file.filename)
    ext = file.filename.split('.')[-1]
    saved_filename = f"{file_id}.{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    
    # Save file
    try:
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Parse file and extract metadata
    try:
        if file_type != 'image':
            df, metadata = parse_spectrum_file(file_path, file_type)
        else:
            metadata = {'type': 'image'}
            df = None
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    # Create dataset info
    dataset_info = DatasetInfo(
        id=file_id,
        name=file.filename,
        type=file_type,
        size=size,
        uploaded_at=datetime.utcnow(),
        metadata=DatasetMetadata(**metadata) if metadata else None,
    )
    
    return UploadResponse(
        success=True,
        message="File uploaded successfully",
        dataset=dataset_info,
    )


@router.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    datasets = []
    
    for filename in os.listdir(settings.UPLOAD_DIR):
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            stat = os.stat(file_path)
            datasets.append({
                "id": filename.split('.')[0],
                "filename": filename,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })
    
    return {"datasets": datasets}


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    # Find file with matching ID
    for filename in os.listdir(settings.UPLOAD_DIR):
        if filename.startswith(dataset_id):
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            os.remove(file_path)
            return {"success": True, "message": "Dataset deleted"}
    
    raise HTTPException(status_code=404, detail="Dataset not found")
