# Use RunPod's official PyTorch base image (CUDA pre-installed)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Copy your requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod

# Copy your OmniVoice folder and handler script
COPY OmniVoice/ /app/OmniVoice/
COPY handler.py /app/

# Tell RunPod what script to run
CMD ["python", "-u", "handler.py"]