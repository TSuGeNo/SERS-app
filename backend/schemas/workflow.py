"""Workflow schemas"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class WorkflowCreate(BaseModel):
    name: str
    description: str
    author: Optional[str] = None
    version: Optional[str] = '1.0'
    category: str
    config: Optional[Dict[str, Any]] = None
    yaml_config: Optional[str] = None

class Workflow(BaseModel):
    id: str
    name: str
    description: str
    author: str
    version: str
    category: str
    config: Dict[str, Any]
    downloads: int = 0
    rating: float = 0

class WorkflowExecuteRequest(BaseModel):
    data: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None

class WorkflowExecuteResponse(BaseModel):
    workflow_id: str
    workflow_name: str
    success: bool = False
    steps_executed: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}
    error: Optional[str] = None
