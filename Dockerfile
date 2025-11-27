# ---------------------------
# Base Image
# ---------------------------
FROM python:3.11-slim

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
# System dependencies (minimal)
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------
# Copy only requirements first (better caching)
# ---------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------
# Copy project files (after dependencies for caching)
# ---------------------------
COPY . .

# ---------------------------
# Expose port (for local use)
# ---------------------------
EXPOSE 8080


# ---------------------------
# Start Uvicorn server
# ---------------------------
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
