# Use a lightweight official Python image
FROM python:3.10-slim

# Install system dependencies required for MediaPipe (including libGLESv2) and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "ai_server.py:app", "--host", "0.0.0.0", "--port", "8000"]
