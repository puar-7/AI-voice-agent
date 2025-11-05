# Start from a stable, slim Python image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - apt-get update: Refreshes package lists
# - libportaudio2: The audio library needed by 'sounddevice'
# - ffmpeg: The audio tool needed by 'pydub'
# - -y: Auto-confirms the installation
# - --no-install-recommends: Installs only essential packages
# - rm -rf /var/lib/apt/lists/*: Cleans up to keep the image small
RUN apt-get update && \
    apt-get install -y libportaudio2 ffmpeg --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
# --no-cache-dir: Disables the pip cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the command to run your application
# This is the same as your "Start Command" in the Render GUI
# It will use the $PORT variable provided by Render
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
