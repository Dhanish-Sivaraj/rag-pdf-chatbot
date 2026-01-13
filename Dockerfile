FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
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

# Create templates directory if it doesn't exist
RUN mkdir -p templates

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=10000

# Expose the port
EXPOSE 10000

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 app:app
