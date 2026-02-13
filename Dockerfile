# PROSPER - Pediatric Pathology Search Tool
# Docker image for easy deployment

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Haoran Chen"
LABEL description="PROSPER: Pediatric Retrieval for Oncology Slides, Pathology Evidence, and Reporting"
LABEL version="1.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # OpenSlide dependencies
    libopenslide0 \
    libopenslide-dev \
    python3-openslide \
    # Image libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install PROSPER in development mode
RUN pip install --no-cache-dir -e .

# Create directories for data and output
RUN mkdir -p /data /output

# Set default command
CMD ["python", "-m", "prosper.cli", "--help"]

# Usage:
# Build: docker build -t prosper:latest .
# Run:   docker run --gpus all -v /path/to/slides:/data -v /path/to/output:/output prosper:latest \
#        python -m prosper.cli --slide-path /data --output-dir /output
