# Dockerfile for Render
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Render
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p templates utils static/uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=10000  # Render uses port 10000
ENV GEMINI_API_KEY=""
ENV SECRET_KEY="your-render-secret-key"

# Expose the port
EXPOSE 10000

# Use gunicorn for production (Render requirement)
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app
