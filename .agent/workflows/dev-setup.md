---
description: Development setup workflow for SERS-Insight Platform
---

# SERS-Insight Platform Development Setup

## Prerequisites
- Node.js 18+ installed
- Python 3.10+ installed
- PostgreSQL installed and running

## Frontend Setup

// turbo
1. Navigate to the frontend directory and install dependencies:
```bash
cd frontend && npm install
```

// turbo
2. Run the development server:
```bash
npm run dev
```

## Backend Setup

// turbo
3. Navigate to the backend directory:
```bash
cd backend
```

4. Create a virtual environment:
```bash
python -m venv venv
```

5. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

// turbo
6. Install Python dependencies:
```bash
pip install -r requirements.txt
```

// turbo
7. Run the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

## Database Setup

8. Create PostgreSQL database:
```sql
CREATE DATABASE sers_insight;
```

9. Run migrations:
```bash
python -m alembic upgrade head
```
