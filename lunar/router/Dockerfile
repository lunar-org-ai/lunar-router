FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/app/data/uniroute_state

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main_local:app", "--host", "0.0.0.0", "--port", "8000"]
