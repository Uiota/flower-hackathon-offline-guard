FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models demo_artifacts

# Set environment variables
ENV OFFLINE_MODE=1
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/api/status || exit 1

# Default command
CMD ["python", "dashboard_with_agents.py"]
