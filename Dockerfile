# ---------------------------
#   Base Python Image
# ---------------------------
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ---------------------------
#   Install System Libraries
# ---------------------------
# Required for FAISS, Whisper, Torch, LangChain, numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
#   Copy project files
# ---------------------------
COPY . .

# ---------------------------
#   Install Python Dependencies
# ---------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
#   Expose FastAPI port
# ---------------------------
EXPOSE 8000

# ---------------------------
#   Start FastAPI server
# ---------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
