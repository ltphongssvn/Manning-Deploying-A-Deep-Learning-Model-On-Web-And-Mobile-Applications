# Food Image Classifier — Web, Browser & Mobile Deep Learning Inference

A full-stack application that classifies food images using a MobileNetV2 deep learning model trained on the Food-101 dataset. Supports three inference paths: **server-side** (FastAPI + TensorFlow), **browser-side** (TensorFlow.js), and **on-device mobile** (React Native + TF.js via Expo development build).

**Live Web Demo:** [https://welcoming-charm-production-52ee.up.railway.app](https://welcoming-charm-production-52ee.up.railway.app)

**Android Dev Build:** [https://expo.dev/accounts/ltphongssvn/projects/mobile](https://expo.dev/accounts/ltphongssvn/projects/mobile)

**Test Dataset:** [https://drive.google.com/drive/folders/1f3PVWNZZyeZhemEXOS1X6u2_5evGoDtJ?usp=sharing](https://drive.google.com/drive/folders/1f3PVWNZZyeZhemEXOS1X6u2_5evGoDtJ?usp=sharing)

## Overview

This project demonstrates an end-to-end machine learning deployment pipeline — from model training on GPU infrastructure (Harvard HPC, NVIDIA A10G) to production deployment on Railway (web) and Expo EAS (mobile). The codebase follows Test-Driven Development (TDD) with 80% per-file coverage enforcement, comprehensive security practices via pre-commit hooks, and GitFlow branching for team collaboration.

### Supported Food Classes

- Apple Pie
- Caesar Salad
- Falafel

## Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                   Mobile App (Milestone 3)                   │
│         React Native + Expo SDK 54 + TF.js (6.8MB)          │
│     On-device inference · Camera/Gallery · No internet       │
│        Android APK (EAS Build) · iOS (pending)               │
├──────────────────────────────────────────────────────────────┤
│                  Web Frontend (Milestone 2)                   │
│              React 19 + TypeScript + Vite 7                  │
│         Browser-side TF.js inference (6.8MB model)           │
├──────────────────────────────────────────────────────────────┤
│                     FastAPI Backend                           │
│        /api/predict  /api/predict_url  /api/classes           │
│         Server-side inference (24MB Keras model)              │
├──────────────────────────────────────────────────────────────┤
│                  Docker (Multi-stage)                         │
│          Node 22 (build) → Python 3.11 (runtime)             │
├──────────────────────────────────────────────────────────────┤
│                Railway (Production)                           │
│            Automatic PORT binding via env var                 │
└──────────────────────────────────────────────────────────────┘
```

## Model Performance

Three models were evaluated using transfer learning on a Food-101 subset (3,000 images, 3 classes) on Harvard HPC (NVIDIA A10G GPU, TensorFlow 2.18):

| Model | Val Accuracy | Training Time | Model Size | Deployed |
|---|---|---|---|---|
| VGG19 | 92.67% | 41s | 83 MB | No |
| ResNet50 | 93.83% | 41s | 115 MB | No |
| **MobileNetV2** | **92.33%** | **44s** | **24 MB** | **Yes** |

MobileNetV2 was selected for deployment due to its optimal balance of accuracy and size. TensorFlow.js conversion with float16 quantization reduces the model to 6.8 MB for browser and mobile inference.

## Tech Stack

**Frontend:** React 19, TypeScript, Vite 7, TensorFlow.js

**Backend:** FastAPI, TensorFlow 2.18, Pillow, uvicorn

**Mobile:** React Native 0.81, Expo SDK 54, TensorFlow.js React Native, EAS Build

**Infrastructure:** Docker (multi-stage), Railway, GitHub Actions CI (5 jobs)

**Testing:** pytest + vitest, 80% per-file coverage enforced, TDD (Red-Green-Refactor)

**Security:** pre-commit/pre-push hooks (14 checks), detect-secrets, SECURITY.md

**Git Workflow:** GitFlow (main → develop → feature branches), PR-based merges

## Project Structure
```
├── backend/
│   ├── app.py                 # FastAPI application (routes, CORS, static serving)
│   ├── inference.py           # Pure inference functions (preprocess, predict)
│   ├── assets/
│   │   ├── classes.json       # Class label mapping
│   │   ├── model_tf/          # Keras model (server inference)
│   │   └── model_tfjs/        # TF.js model (browser + mobile inference)
│   ├── test_api.py            # API integration tests (28 tests)
│   └── test_e2e.py            # E2E tests with real TF model (12 tests)
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main UI component (dual inference)
│   │   ├── App.test.tsx       # Component tests (27 tests)
│   │   └── test-setup.ts      # Vitest + jsdom setup
│   ├── package.json
│   └── vite.config.ts         # Vitest config with 80% coverage thresholds
├── mobile/
│   ├── App.tsx                # Root component (bottom tab navigation)
│   ├── HomeScreen.tsx         # Camera/gallery + on-device TF.js inference
│   ├── AboutScreen.tsx        # App info screen
│   ├── ModelService.ts        # TF.js model loading + classification logic
│   ├── config.tsx             # App-wide constants (image size, classes)
│   ├── metro.config.js        # Metro bundler config (.bin asset support)
│   ├── eas.json               # EAS Build profiles (dev/preview/production)
│   ├── app.json               # Expo app config with EAS project ID
│   └── assets/model_tfjs/     # Bundled TF.js model (offline inference)
├── .github/
│   └── workflows/ci.yml       # 5-job CI pipeline
├── .pre-commit-config.yaml    # 14 hooks (pre-commit + pre-push)
├── .secrets.baseline          # detect-secrets baseline (clean)
├── Dockerfile                 # Multi-stage (Node build → Python runtime)
├── docker-compose.yml         # Local development
├── pyproject.toml             # Python deps, pytest config, coverage thresholds
├── SECURITY.md                # Security practices documentation
└── README.md
```

## Test-Driven Development (TDD)

This project implements TDD following the Red-Green-Refactor cycle with Given/When/Then test structure, ZOMBIES mnemonic for edge cases, and surgical refactoring for testability.

### TDD Principles Applied

**Test Design:** Every test follows Given/When/Then structure with clear intent. Tests are fast (in-memory), independent, and repeatable. No branching logic or loops in test code.

**Test Doubles:** FakeModel (substitute TF model returning deterministic predictions), mocked fetch/TF.js APIs, stubbed HTTP responses for URL prediction tests. Production code uses dependency injection via FastAPI's `app.state.model` for easy test double substitution.

**Surgical Refactoring:** `backend/inference.py` extracts pure functions (`preprocess_image`, `run_prediction`) from `app.py` — making inference logic testable without a running server. Lifespan handler isolated with `# pragma: no cover` for infrastructure code.

**ZOMBIES Coverage:** Zero (empty inputs), One (single prediction), Many (batch), Boundaries (10x10, 2000x2000 images), Interfaces (API contract), Exceptions (invalid files, network errors), Simple (happy paths).

### Test Suite Summary

| Layer | Tests | Runtime | Trigger | Framework |
|---|---|---|---|---|
| Backend unit (inference) | 15 | 0.6s | pre-commit, CI | pytest |
| Backend API integration | 13 | 0.6s | pre-commit, CI | pytest + httpx |
| Frontend component | 27 | 4.2s | pre-commit, CI | vitest + jsdom |
| E2E real model | 12 | 7.5s | CI only | pytest + real TF |
| **Total** | **67** | **~13s** | | |

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

### Coverage Enforcement (80% Per-File Minimum)

Coverage thresholds follow industry standards and are enforced at three levels: pre-push hooks, CI pipeline, and tool configuration. No file is allowed below 80% on any metric.

**Industry Standard Minimums Enforced:**

| Metric | Threshold | Enforcement |
|---|---|---|
| Line coverage | 80% per file | pytest `fail_under=80`, vitest `perFile: true` |
| Branch coverage | 80% per file | pytest `branch=true`, vitest `branches: 80` |
| Function coverage | 80% per file | vitest `functions: 80` |
| Statement coverage | 80% per file | vitest `statements: 80` |

**Current Coverage (all above 80%):**

| File | Lines | Branch | Functions | Statements |
|---|---|---|---|---|
| `backend/app.py` | 100% | 100% | 100% | 100% |
| `backend/inference.py` | 100% | 100% | 100% | 100% |
| `frontend/src/App.tsx` | 100% | 82.1% | 100% | 97.9% |

**Backend Configuration** (`pyproject.toml`):
```toml
[tool.coverage.run]
source = ["backend"]
branch = true

[tool.coverage.report]
fail_under = 80
show_missing = true
```

**Frontend Configuration** (`frontend/vite.config.ts`):
```typescript
coverage: {
  provider: 'v8',
  thresholds: {
    perFile: true,
    lines: 80,
    branches: 80,
    functions: 80,
    statements: 80,
  },
}
```

## Pre-commit and Pre-push Hooks

14 automated checks run on every commit and push to enforce code quality, security, and test coverage before code reaches the remote repository.

### Pre-commit Hooks (run on every `git commit`)

| Hook | Purpose |
|---|---|
| detect-secrets | Scan staged files against `.secrets.baseline` for leaked credentials |
| block-large-files | Reject `.h5`, `.pkl`, `.pth`, `.onnx` binary files |
| check-env-files | Prevent `.env` files from being committed |
| verify-selective-staging | Display staged files for review |
| mixed-line-ending | Enforce LF line endings (no CRLF) |
| end-of-file-fixer | Ensure files end with newline |
| trailing-whitespace | Remove trailing whitespace |
| check-json | Validate JSON syntax |
| check-yaml | Validate YAML syntax |
| check-added-large-files | Block files over 500KB |
| pytest-quick | Run backend tests (fast feedback) |
| vitest-quick | Run frontend tests (fast feedback) |

### Pre-push Hooks (run on every `git push`)

| Hook | Purpose |
|---|---|
| pytest-coverage-push | Backend tests with 80% per-file coverage enforcement |
| vitest-coverage-push | Frontend tests with 80% per-file coverage enforcement |

### How It Works

Every `git commit` triggers 12 pre-commit checks in under 10 seconds. Every `git push` additionally runs coverage-enforced test suites. If any check fails, the commit/push is rejected and the developer must fix the issue before proceeding. This ensures no untested or insecure code reaches the remote repository.

## CI/CD Pipeline

GitHub Actions runs 5 parallel jobs on every push to any branch:

| Job | What It Does | Depends On |
|---|---|---|
| **backend-tests** | pytest with coverage enforcement (80% per-file) | — |
| **frontend-tests** | vitest with coverage enforcement (80% per-file) | — |
| **e2e-tests** | Real TensorFlow model integration tests (12 tests) | — |
| **security-scan** | detect-secrets audit against baseline | — |
| **docker-build** | Full Docker image build verification | all 4 above |

The pipeline enforces that all tests pass, coverage thresholds are met, no secrets are leaked, and the Docker image builds successfully before any deployment.

## Security

11 security layers are implemented and documented in [SECURITY.md](SECURITY.md):

| Layer | Implementation |
|---|---|
| Secret Detection | detect-secrets with `.secrets.baseline` |
| Binary Protection | Pre-commit blocks `.h5`, `.pkl`, `.pth`, `.onnx` |
| Environment Protection | Pre-commit blocks `.env` files |
| File Size Limits | 500KB maximum per file |
| Dependency Scanning | npm audit, pip audit in CI |
| CORS Configuration | FastAPI middleware with configurable origins |
| Input Validation | File type checking, URL validation, size limits |
| Error Handling | Graceful 400 errors, no stack traces in production |
| TF Log Suppression | `TF_CPP_MIN_LOG_LEVEL=3` hides internal paths |
| Dynamic PORT | Railway env var, no hardcoded ports |
| HTTPS | Railway provides TLS termination |

## Mobile App (Milestone 3)

The mobile app runs the same MobileNetV2 model entirely on-device using TensorFlow.js React Native — no internet connection required for inference.

### Mobile Tech Stack

| Component | Version | Purpose |
|---|---|---|
| React Native | 0.81.5 | Cross-platform native UI |
| Expo SDK | 54 | Build toolchain and native modules |
| TensorFlow.js | 4.22 | On-device ML inference |
| @tensorflow/tfjs-react-native | 1.0 | React Native TF.js bindings |
| expo-image-picker | 17.0 | Camera and gallery access |
| expo-gl | 16.0 | WebGL backend for TF.js |
| @react-navigation/bottom-tabs | 7.x | Tab navigation (Home/About) |
| EAS Build | 18.x | Cloud-based native builds |

### Development Build vs Expo Go

This project uses **Expo Development Builds** (not Expo Go) for full native module support:

| Feature | Expo Go | Development Build |
|---|---|---|
| Custom native modules | Limited | Full support |
| TF.js GL backend | Version constrained | Native compilation |
| App distribution | Expo Go required | Standalone APK/IPA |
| Offline capability | Requires Expo Go | Fully standalone |
| Production path | Not possible | EAS Submit to stores |

### Build Profiles (eas.json)

| Profile | Purpose | Distribution |
|---|---|---|
| `development` | Dev client with hot reload | Internal (APK/IPA) |
| `preview` | Testing build without dev tools | Internal (APK/IPA) |
| `production` | Store-ready build | App Store / Google Play |

### Running the Mobile App
```bash
# Install dependencies
cd mobile && npm install

# Start Metro dev server
REACT_NATIVE_PACKAGER_HOSTNAME=<YOUR_WIFI_IP> npx expo start --dev-client

# Build Android APK
eas build --platform android --profile development

# Build iOS (requires Apple Developer account)
eas build --platform ios --profile development
```

### Mobile App Features

- On-device TF.js inference (no server required, works offline)
- Camera and photo gallery image selection with permission handling
- Real-time prediction display with confidence scores and inference timing
- Dark theme UI with flex layout optimized for all screen sizes
- Bottom tab navigation (Home / About)
- Model warm-up on startup for faster first prediction

## Git Workflow (GitFlow)

This project follows GitFlow branching for professional team collaboration:
```
main (production) ← develop (integration) ← feature/* (work)
```

| Branch | Purpose | Merges To |
|---|---|---|
| `main` | Production-ready code, deployed to Railway | — |
| `develop` | Integration branch, all features merge here first | `main` (via PR) |
| `feature/*` | Individual feature work | `develop` (via PR) |

PRs require all CI checks to pass. Solo projects merge without approval gate.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 22+
- Docker (optional, for containerized runs)
- Expo Go or EAS Build (for mobile development)

### Web App (Local Development)
```bash
git clone https://github.com/ltphongssvn/Manning-Deploying-A-Deep-Learning-Model-On-Web-And-Mobile-Applications.git
cd Manning-Deploying-A-Deep-Learning-Model-On-Web-And-Mobile-Applications

# Backend
pip install uv && uv sync --group dev
uv run uvicorn backend.app:app --reload

# Frontend (separate terminal)
cd frontend && npm ci && npm run dev
```

### Docker
```bash
docker compose up --build
# Access at http://localhost:8000
```

### Deploy to Railway
```bash
railway link && railway up
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload image file for classification |
| `POST` | `/api/predict_url` | Provide image URL for classification |
| `GET` | `/api/classes` | List supported food classes |
| `GET` | `/artifacts/model_tfjs/model.json` | TF.js model for browser inference |

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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (Railway sets this automatically) |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppress TensorFlow C++ logs |
| `VITE_API_URL` | `""` | API base URL for frontend (empty = same origin) |

## License

This project is part of the Manning liveProject series: *Deploying a Deep Learning Model on Web and Mobile Applications Using TensorFlow*.
