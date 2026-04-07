# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system build tools for compiling packages like numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libffi-dev \
        libssl-dev \
        wget \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (leverage Docker cache)
COPY requirements.txt .
COPY runtime.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port (adjust for Streamlit)
EXPOSE 8501

# Command to run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]