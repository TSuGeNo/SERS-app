# SERS-Insight Platform

A modular, extensible web-based platform for Surface-Enhanced Raman Spectroscopy (SERS) analysis with intelligent framework selection, automated modelling, and community-driven workflows.

![SERS-Insight Platform](./docs/screenshot.png)

## Features

### 🔬 LSPR Simulation
- Drude-Lorentz model for Ag/Au nanoparticles
- Mie theory-based enhancement factor prediction
- Material and shape optimization recommendations

### 🧪 Molecule Detection
- Reference peak library for common SERS probes (R6G, CV, NB, MB)
- Automatic peak matching with confidence scoring
- Concentration regression for quantification

### 🧬 Biomolecule Classification
- PCA dimensionality reduction
- SVM and Random Forest classifiers
- Cross-validation with detailed metrics

### 🦠 Pathogen Detection
- 1D CNN architecture for bacterial classification
- Data augmentation for improved accuracy
- Ensemble methods (CNN + SVM)

### 📊 Visualization Studio
- Interactive Plotly charts
- Publication-ready exports (PNG, SVG)
- Multiple chart types (spectrum, PCA, heatmap, confusion matrix)

### 🔄 Custom Workflows
- YAML-based workflow definitions
- Community workflow marketplace
- Fork and customize existing workflows

### 💬 AI-Powered Chat
- Julius AI-like interface
- Natural language data analysis
- Context-aware recommendations

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Accessible UI components
- **Zustand** - State management
- **Plotly.js** - Interactive visualizations

### Backend
- **FastAPI** - Modern Python API framework
- **scikit-learn** - Machine learning
- **NumPy/SciPy** - Scientific computing
- **Pandas** - Data manipulation
- **PostgreSQL** - Database

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- PostgreSQL (optional, for production)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at http://localhost:3000

### Backend Setup

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API will be available at http://localhost:8000
API docs at http://localhost:8000/docs

## Project Structure

```
sers-insight/
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   ├── components/      # React components
│   │   │   ├── chat/        # Chat interface
│   │   │   ├── layout/      # Layout components
│   │   │   ├── upload/      # File upload
│   │   │   └── ui/          # shadcn/ui components
│   │   └── lib/
│   │       └── stores/      # Zustand stores
│   └── package.json
│
├── backend/                  # FastAPI backend
│   ├── api/
│   │   └── routes/          # API endpoints
│   ├── core/                # Configuration
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   └── main.py
│
└── docs/                    # Documentation
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload data files |
| POST | `/api/preprocess` | Apply preprocessing pipeline |
| POST | `/api/analyze` | Run analysis framework |
| POST | `/api/simulate` | Run LSPR simulation |
| GET | `/api/workflows` | List available workflows |
| POST | `/api/chat` | Send chat message |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Inspired by [Julius AI](https://julius.ai)
- SERS reference data from literature
