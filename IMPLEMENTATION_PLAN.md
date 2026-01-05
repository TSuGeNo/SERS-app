# SERS-Insight Platform - Implementation Plan

## Overview

A modular, extensible web-based platform for Surface-Enhanced Raman Spectroscopy (SERS) analysis, featuring:
- **Julius AI-like Interface**: Chat-based data interaction
- **SERS-Specific Analysis**: Simulation, molecule detection, pathogen classification
- **Custom Workflows**: User-contributed analysis pipelines
- **AI-Powered Insights**: Automated framework selection and analysis

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **UI Library**: shadcn/ui + Tailwind CSS
- **Charts**: Plotly.js, Recharts
- **State Management**: Zustand
- **File Upload**: react-dropzone

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **ML/DL**: scikit-learn, TensorFlow, PyTorch
- **Scientific Computing**: NumPy, SciPy, pandas
- **SERS Processing**: Custom physics modules
- **AI Integration**: OpenAI/Anthropic API for chat insights

### Database
- **Primary**: PostgreSQL
- **File Storage**: Local filesystem (S3-compatible for production)

---

## Project Structure

```
sers-insight/
├── frontend/                    # Next.js application
│   ├── app/
│   │   ├── (auth)/             # Authentication pages
│   │   ├── (dashboard)/        # Main application
│   │   │   ├── page.tsx        # Dashboard home
│   │   │   ├── analyze/        # Analysis interface
│   │   │   ├── simulate/       # LSPR simulation
│   │   │   ├── workflows/      # Community workflows
│   │   │   └── history/        # Analysis history
│   │   ├── api/                # API routes
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ui/                 # shadcn/ui components
│   │   ├── chat/               # Chat interface components
│   │   ├── upload/             # File upload components
│   │   ├── visualization/      # Chart components
│   │   └── analysis/           # Analysis-specific components
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   ├── utils.ts            # Utilities
│   │   └── stores/             # Zustand stores
│   └── types/                  # TypeScript types
│
├── backend/                    # FastAPI application
│   ├── main.py                 # Entry point
│   ├── api/
│   │   ├── routes/             # API endpoints
│   │   ├── deps.py             # Dependencies
│   │   └── middleware.py       # Middleware
│   ├── core/
│   │   ├── config.py           # Configuration
│   │   ├── security.py         # Auth utilities
│   │   └── database.py         # Database connection
│   ├── models/                 # SQLAlchemy models
│   ├── schemas/                # Pydantic schemas
│   ├── services/
│   │   ├── preprocessing/      # Data preprocessing
│   │   ├── simulation/         # LSPR simulation
│   │   ├── detection/          # Molecule detection
│   │   ├── classification/     # ML classification
│   │   ├── cnn/                # Deep learning
│   │   ├── workflow/           # Custom workflows
│   │   ├── visualization/      # Plot generation
│   │   └── ai/                 # AI chat integration
│   ├── data/
│   │   ├── reference_spectra/  # Reference libraries
│   │   └── models/             # Trained models
│   └── tests/
│
└── docs/                       # Documentation
```

---

## Phase 1: Core Infrastructure (Current Phase)

### Tasks
1. ✅ Project structure setup
2. ⬜ Next.js frontend with shadcn/ui
3. ⬜ FastAPI backend skeleton
4. ⬜ Database models
5. ⬜ Basic file upload
6. ⬜ Chat interface UI

---

## Phase 2: Data Processing Pipeline

### Tasks
1. ⬜ CSV/TXT/XLSX file parsing
2. ⬜ Baseline correction (ALS algorithm)
3. ⬜ Savitzky-Golay smoothing
4. ⬜ Normalization methods
5. ⬜ Peak detection
6. ⬜ Preprocessing visualization

---

## Phase 3: SERS Simulation Module

### Tasks
1. ⬜ Drude-Lorentz dielectric model
2. ⬜ Mie theory implementation
3. ⬜ Enhancement factor calculator
4. ⬜ Material comparison (Ag/Au)
5. ⬜ Interactive simulation UI

---

## Phase 4: Analysis Frameworks

### Molecule Detection
1. ⬜ Peak library system
2. ⬜ R6G detection
3. ⬜ Concentration regression

### Biomolecule Identification
1. ⬜ PCA implementation
2. ⬜ SVM/RF classifiers
3. ⬜ Cross-validation

### Pathogen Detection
1. ⬜ 1D CNN architecture
2. ⬜ Training pipeline
3. ⬜ Model inference

---

## Phase 5: AI Chat Integration

### Tasks
1. ⬜ Chat UI component
2. ⬜ AI backend integration
3. ⬜ Context-aware responses
4. ⬜ Analysis commands
5. ⬜ Insights generation

---

## Phase 6: Workflow System

### Tasks
1. ⬜ YAML workflow parser
2. ⬜ Workflow execution engine
3. ⬜ Community workflow storage
4. ⬜ Workflow marketplace UI

---

## API Endpoints Overview

### Data Management
- `POST /api/upload` - Upload data files
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}` - Get dataset details
- `DELETE /api/datasets/{id}` - Delete dataset

### Preprocessing
- `POST /api/preprocess` - Apply preprocessing pipeline
- `GET /api/preprocess/options` - Get available methods

### Analysis
- `POST /api/analyze` - Run analysis
- `POST /api/simulate` - Run LSPR simulation
- `GET /api/frameworks` - List available frameworks

### Workflows
- `GET /api/workflows` - List workflows
- `POST /api/workflows` - Create workflow
- `POST /api/workflows/{id}/execute` - Execute workflow

### Chat
- `POST /api/chat` - Send chat message
- `GET /api/chat/history` - Get chat history

### Visualization
- `GET /api/visualizations/{id}` - Get visualization data
- `POST /api/visualizations/export` - Export plots
