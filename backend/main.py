"""
SERS-Insight Platform - FastAPI Backend
Main entry point for the API server
"""


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api.routes import (
    health,
    upload,
    preprocess,
    analyze,
    simulate,
    workflows,
    chat,
    visualizations,
)
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting SERS-Insight Platform API...")
    
    # Create upload directories if they don't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    
    yield
    
    # Shutdown
    print("👋 Shutting down SERS-Insight Platform API...")


app = FastAPI(
    title="SERS-Insight Platform API",
    description="""
    A modular, extensible API for Surface-Enhanced Raman Spectroscopy (SERS) analysis 
    with intelligent framework selection, automated modelling, and community-driven workflows.
    
    ## Features
    
    * **Data Upload & Preprocessing**: CSV, TXT, XLSX file support with baseline correction, smoothing, normalization
    * **LSPR Simulation**: Drude-Lorentz model for Ag/Au nanoparticle enhancement prediction
    * **Molecule Detection**: Peak matching and concentration regression for R6G and other SERS probes
    * **Classification**: PCA + SVM/RF for biomolecule identification, CNN for pathogen detection
    * **Spectral Unmixing**: NMF and ICA for mixed sample analysis
    * **Custom Workflows**: YAML-based workflow definition and execution
    * **AI Chat**: Interactive data analysis with LLM integration
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for results/visualizations
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(preprocess.router, prefix="/api", tags=["Preprocessing"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(simulate.router, prefix="/api", tags=["Simulation"])
app.include_router(workflows.router, prefix="/api", tags=["Workflows"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(visualizations.router, prefix="/api", tags=["Visualizations"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SERS-Insight Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
