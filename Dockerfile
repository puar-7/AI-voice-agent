# Start from a stable, slim Python image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    libportaudio2 \
    ffmpeg \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Verify edge-tts is installed and accessible
RUN edge-tts --version || echo "edge-tts CLI not found in PATH"

# Copy the rest of your application code
COPY . .

# Set environment variables for edge-tts
ENV PYTHONUNBUFFERED=1
ENV USE_CLI_METHOD=true

# Expose port (optional, for documentation)
EXPOSE 8000

# Set the command to run your application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
