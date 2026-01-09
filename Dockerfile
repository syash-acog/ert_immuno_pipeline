FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    procps \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy tools directory (Using local copy as requested)
COPY tools /app/tools

# Add Tools and src to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/tools/NetCleave:/app/tools/ImmuScope:/app/src"

# Copy source code
COPY src /app/src
COPY data /app/data


CMD ["python", "/app/src/main.py"]
