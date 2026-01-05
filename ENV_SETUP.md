# SERS-Insight Platform - Environment Setup

## OpenRouter API Integration

This platform uses **OpenRouter** for unified access to multiple AI models. With a single API key, you can access:

| Model ID | Display Name | Provider |
|----------|--------------|----------|
| `openai/gpt-5-mini` | GPT-5 Mini | OpenAI |
| `anthropic/claude-sonnet-4.5` | Claude Sonnet 4.5 | Anthropic |
| `google/gemini-3-flash-preview` | Gemini 3 Flash Preview | Google |

## Backend Configuration

1. Create a `.env` file in the `backend/` directory:

```bash
cd backend
copy .env.example .env
```

2. Add your OpenRouter API key:

```env
# SERS-Insight Platform Backend Configuration

# API Settings
HOST=0.0.0.0
PORT=8000
DEBUG=True
SECRET_KEY=your-secret-key-change-in-production

# File Storage
UPLOAD_DIR=uploads
RESULTS_DIR=results
MODELS_DIR=models
MAX_UPLOAD_SIZE=52428800

# OpenRouter API Integration
# Get your API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key

# Default AI model to use
DEFAULT_AI_MODEL=anthropic/claude-sonnet-4.5
```

## Frontend Configuration

Create a `.env.local` file in the `frontend/` directory:

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Starting the Application

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

## Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

## Getting Your OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up or log in
3. Navigate to [API Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Add the key to your `.env` file

## Supported Features

- **Multi-Model Chat**: Switch between GPT-5, Claude, and Gemini
- **SERS Analysis**: Peak detection, molecule identification
- **Data Upload**: CSV, TXT, JSON spectral data
- **Code Generation**: Python analysis scripts
- **Visualization**: Real-time spectrum plots
