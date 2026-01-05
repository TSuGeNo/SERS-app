"""Upload schemas"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DatasetMetadata(BaseModel):
    columns: Optional[List[str]] = None
    rows: Optional[int] = None
    wavenumber_column: Optional[str] = None
    wavenumber_range: Optional[List[float]] = None
    has_headers: Optional[bool] = None

class DatasetInfo(BaseModel):
    id: str
    name: str
    type: str
    size: int
    uploaded_at: datetime
    metadata: Optional[DatasetMetadata] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    dataset: DatasetInfo
