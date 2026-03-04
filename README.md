# Food Image Classifier — Browser & Server-based Deep Learning Inference

A full-stack web application that classifies food images using a MobileNetV2 deep learning model trained on the Food-101 dataset. Supports both **server-side inference** (FastAPI + TensorFlow) and **client-side inference** (TensorFlow.js in the browser).

**Live Demo:** [https://welcoming-charm-production-52ee.up.railway.app](https://welcoming-charm-production-52ee.up.railway.app)

**Test Dataset:** [https://drive.google.com/drive/folders/1f3PVWNZZyeZhemEXOS1X6u2_5evGoDtJ?usp=sharing]
## Overview

This project demonstrates an end-to-end machine learning deployment pipeline — from model training on GPU infrastructure to production deployment on Railway. Users can upload food images or provide URLs, and the application returns classification predictions with confidence scores through both server and browser inference paths simultaneously.

### Supported Food Classes

- Apple Pie
- Caesar Salad
- Falafel

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React + TypeScript                   │
│              (Vite build, TensorFlow.js)                │
│         Browser-side inference (6.8MB model)            │
├─────────────────────────────────────────────────────────┤
│                     FastAPI Backend                     │
│        /api/predict  /api/predict_url  /api/classes     │
│         Server-side inference (24MB Keras model)        │
├─────────────────────────────────────────────────────────┤
│                  Docker (Multi-stage)                   │
│          Node 22 (build) → Python 3.11 (runtime)        │
├─────────────────────────────────────────────────────────┤
│                Railway (Production)                     │
│            Automatic PORT binding via env var           │
└─────────────────────────────────────────────────────────┘
```

## Model Performance

Three models were evaluated using transfer learning on a Food-101 subset (3,000 images, 3 classes):

| Model | Val Accuracy | Training Time | Model Size | Deployed |
|---|---|---|---|---|
| VGG19 | 92.67% | 41s | 83 MB | No |
| ResNet50 | 93.83% | 41s | 115 MB | No |
| **MobileNetV2** | **92.33%** | **44s** | **24 MB** | **Yes** |

MobileNetV2 was selected for deployment due to its optimal balance of accuracy and size. The TensorFlow.js conversion with float16 quantization further reduces the browser model to 6.8 MB.

## Tech Stack

**Frontend:** React 19, TypeScript, Vite 7, TensorFlow.js

**Backend:** FastAPI, TensorFlow 2.18, Pillow, uvicorn

**Infrastructure:** Docker (multi-stage), Railway, GitHub Actions CI

**Testing:** pytest (backend), vitest (frontend), 80% per-file coverage enforced

**Security:** pre-commit hooks, detect-secrets, automated secret scanning

## Project Structure

```
├── backend/
│   ├── app.py                 # FastAPI application (routes, CORS, static serving)
│   ├── inference.py           # Pure inference functions (preprocess, predict)
│   ├── assets/
│   │   ├── classes.json       # Class label mapping
│   │   ├── model_tf/          # Keras model (server inference)
│   │   └── model_tfjs/        # TF.js model (browser inference)
│   ├── test_api.py            # API integration tests (28 tests)
│   └── test_e2e.py            # E2E tests with real TF model (12 tests)
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main UI component (dual inference)
│   │   ├── App.test.tsx       # Component tests (27 tests)
│   │   └── test-setup.ts      # Vitest + jsdom setup
│   ├── package.json
│   └── vite.config.ts         # Vitest config with 80% coverage thresholds
├── .github/
│   └── workflows/ci.yml       # 5-job CI pipeline
├── .pre-commit-config.yaml    # 14 hooks (pre-commit + pre-push)
├── .secrets.baseline          # detect-secrets baseline
├── Dockerfile                 # Multi-stage (Node build → Python runtime)
├── docker-compose.yml         # Local development
├── pyproject.toml             # Python deps, pytest config, coverage thresholds
├── SECURITY.md                # Security practices documentation
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 22+
- Docker (optional, for containerized runs)

### Local Development

```bash
# Clone the repository
git clone https://github.com/ltphongssvn/Manning-Deploying-A-Deep-Learning-Model-On-Web-And-Mobile-Applications.git
cd Manning-Deploying-A-Deep-Learning-Model-On-Web-And-Mobile-Applications

# Install backend dependencies
pip install uv
uv sync --group dev

# Install frontend dependencies
cd frontend && npm ci && cd ..

# Run backend (starts on port 8000)
uv run uvicorn backend.app:app --reload

# Run frontend dev server (in separate terminal)
cd frontend && npm run dev
```

### Docker

```bash
# Build and run
docker compose up --build

# Access at http://localhost:8000
```

### Deploy to Railway

```bash
# Link to Railway project
railway link

# Deploy
railway up
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload image file for classification |
| `POST` | `/api/predict_url` | Provide image URL for classification |
| `GET` | `/api/classes` | List supported food classes |
| `GET` | `/artifacts/model_tfjs/model.json` | TF.js model for browser inference |

### Example: File Upload

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@food_image.jpg"
```

### Example: URL Prediction

```bash
curl -X POST http://localhost:8000/api/predict_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/food.jpg"}'
```

### Response Format

```json
{
  "predictions": [
    {"class": "falafel", "probability": 0.996},
    {"class": "apple_pie", "probability": 0.002},
    {"class": "caesar_salad", "probability": 0.002}
  ],
  "inference_time_ms": 146.91
}
```

## Testing

### Test Suite Summary

| Layer | Tests | Runtime | Trigger |
|---|---|---|---|
| Backend unit (inference) | 15 | 0.6s | pre-commit, CI |
| Backend API integration | 13 | 0.6s | pre-commit, CI |
| Frontend component | 27 | 4.2s | pre-commit, CI |
| E2E real model | 12 | 7.5s | CI only |
| **Total** | **67** | **~13s** | |

### Running Tests

```bash
# Backend unit + integration tests (excludes E2E by default)
uv run pytest -v

# E2E tests with real TensorFlow model
uv run pytest -m e2e -v

# Backend with coverage report
uv run pytest --cov=backend --cov-report=term-missing --cov-branch

# Frontend tests
cd frontend && npx vitest run

# Frontend with coverage
cd frontend && npx vitest run --coverage
```

### Coverage Thresholds (Enforced)

All files must maintain 80% minimum coverage across lines, branches, functions, and statements. This is enforced at three levels: pre-push hooks, CI pipeline, and coverage configuration.

| File | Lines | Branch | Threshold |
|---|---|---|---|
| `backend/app.py` | 100% | 100% | 80% |
| `backend/inference.py` | 100% | 100% | 80% |
| `frontend/App.tsx` | 100% | 82.1% | 80% |

## CI/CD Pipeline

GitHub Actions runs 5 parallel jobs on every push:

1. **backend-tests** — pytest with coverage enforcement
2. **frontend-tests** — vitest with coverage enforcement
3. **e2e-tests** — Real TensorFlow model integration tests
4. **security-scan** — detect-secrets audit
5. **docker-build** — Full Docker image build verification (runs after all tests pass)

## Security

Pre-commit hooks enforce 14 checks on every commit and push, including secret detection (detect-secrets), large binary blocking, environment file protection, and test execution. See [SECURITY.md](SECURITY.md) for the full security practices documentation.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (Railway sets this automatically) |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppress TensorFlow C++ logs |
| `VITE_API_URL` | `""` | API base URL for frontend (empty = same origin) |

## License

This project is part of the Manning liveProject series: *Deploying a Deep Learning Model on Web and Mobile Applications Using TensorFlow*.
