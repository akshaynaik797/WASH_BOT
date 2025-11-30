# ---------------------------
# Base Image
# ---------------------------
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run expects this variable
ENV PORT=8080

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# System dependencies (FAISS requires these)
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Copy only requirements first (better caching)
# ---------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Copy project files
# ---------------------------
COPY . .

# ---------------------------
# Health check (important for Cloud Run)
# ---------------------------
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ---------------------------
# Start Uvicorn server with longer timeout
# ---------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]

