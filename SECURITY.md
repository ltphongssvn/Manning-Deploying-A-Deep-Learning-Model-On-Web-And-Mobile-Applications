# SECURITY.md
# Security Best Practices Implementation
# Project: Manning Deploying A Deep Learning Model On Web And Mobile Applications
# Stack: Python (FastAPI + TensorFlow) backend, React (Vite + TF.js) frontend, Docker deployment

## Secret Management Implementation

### 1. Pre-commit Secret Detection Setup

**Installation:**
```bash
pip install pre-commit detect-secrets
```

**Configuration (.pre-commit-config.yaml):**
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: .*\.lock|.*\.log|package-lock\.json|uv\.lock
```

**Activation:**
```bash
detect-secrets scan > .secrets.baseline
pre-commit install
pre-commit install --hook-type pre-push
```

### 2. Environment Variables Required

Create `.env` file (never commit):
```bash
# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=3

# Railway Deployment
PORT=8000
RAILWAY_ENVIRONMENT=production

# API Configuration (if adding auth later)
# API_SECRET_KEY=your_secret_key_here  # pragma: allowlist secret
```

### 3. Security Measures Implemented

| Layer | Tool/Practice | Purpose |
|-------|--------------|---------|
| Secret Detection | detect-secrets + pre-commit | Prevent secrets from entering codebase |
| .env Protection | pre-commit hook `check-env-files` | Block .env files from being committed |
| Binary Blocking | pre-commit hook `block-large-files` | Prevent large model files (.h5, .pkl, .pth) from accidental commit |
| Python Dependencies | `uv audit` / pip-audit | Scan for vulnerable Python packages |
| JS Dependencies | `npm audit` | Scan for vulnerable npm packages |
| CORS | FastAPI CORSMiddleware | Restrict cross-origin requests (currently open for development) |
| Input Validation | FastAPI type hints + Pydantic | Validate all request payloads |
| File Upload Safety | PIL.Image.open() validation | Only accept valid image files |
| Docker Isolation | Multi-stage Dockerfile | Minimal production image, no dev tools |
| Model Security | .gitignore + .dockerignore | Control model file access and distribution |
| Type Safety | TypeScript strict mode (frontend) | Catch errors at compile time |
| Test Coverage | pytest + vitest (80% per-file min) | Prevent regressions, enforced via pre-push hooks |

### 4. FastAPI-Specific Security Notes

- **CORS:** Currently `allow_origins=["*"]` for development. Restrict to specific domains in production
- **File Uploads:** `POST /api/predict` accepts image files — validated by PIL before processing
- **No Auth Required:** This is a demo classifier; add JWT/OAuth if deploying for real users
- **Model Loading:** TF model loaded at startup via lifespan handler, not at import time (prevents import-time side effects)
- **Static Files:** TF.js model served via `/artifacts/` — browser downloads model for client-side inference

### 5. Pre-commit Workflow

Every commit automatically:
1. Scans for secrets using detect-secrets
2. Blocks commit if new secrets found
3. Blocks .env files from being committed
4. Blocks large binary files (.h5, .pkl, .pth, .onnx)
5. Validates JSON and YAML files
6. Fixes line endings and trailing whitespace
7. Runs pytest (backend) quick tests
8. Runs vitest (frontend) quick tests

Every push automatically:
1. Runs pytest with coverage enforcement (80% per-file minimum)
2. Runs vitest with coverage enforcement (80% per-file minimum)
3. Runs Docker build verification

### 6. GitHub Actions CI

On every push/merge to main:
1. Backend: pytest with coverage report
2. Frontend: vitest with coverage report
3. Docker: build verification
4. Security: detect-secrets scan

### 7. Team Guidelines

- Never commit `.env` files
- Use environment variables for all credentials
- Run `pre-commit install && pre-commit install --hook-type pre-push` after cloning
- Review `.secrets.baseline` changes carefully
- Rotate any accidentally exposed keys immediately
- Run `uv audit` and `npm audit` regularly
- Keep all dependencies updated
- Restrict CORS origins before production deployment
- Never expose TF model training data or private datasets

### 8. Dependency Security Audit
```bash
# Python: check for known vulnerabilities
pip-audit

# JavaScript: check for known vulnerabilities
cd frontend && npm audit

# Fix automatically where possible
npm audit fix
```

## Verification
```bash
# Run all pre-commit hooks
$ pre-commit run --all-files

# Expected output:
Detect secrets...........................................................Passed
Block Large Binary Files.................................................Passed
Block .env Files.........................................................Passed
Verify Selective Staging.................................................Passed
Fix Line Endings to LF..................................................Passed
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
Validate JSON Files......................................................Passed
Validate YAML Files......................................................Passed
Check for Large Files....................................................Passed
Pytest Quick Tests (Backend).............................................Passed
Vitest Quick Tests (Frontend)............................................Passed
```

All secrets removed and pre-commit/pre-push protection active.
