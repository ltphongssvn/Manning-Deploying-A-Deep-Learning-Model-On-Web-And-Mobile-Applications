FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist
COPY backend/ /app/backend/
COPY pyproject.toml uv.lock .python-version /app/
RUN pip install --no-cache-dir tensorflow==2.18.0 fastapi uvicorn[standard] python-multipart aiofiles pillow numpy requests jinja2
ENV TF_CPP_MIN_LOG_LEVEL=3
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
