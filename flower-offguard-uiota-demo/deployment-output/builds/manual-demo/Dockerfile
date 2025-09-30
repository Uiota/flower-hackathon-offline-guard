FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV OFFLINE_MODE=1
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x run_server.sh run_clients.sh demo_basic.py

# Create artifacts directory
RUN mkdir -p artifacts demo_artifacts

# Expose port for FL server
EXPOSE 8080

# Default command runs the basic demo
CMD ["python3", "demo_basic.py"]