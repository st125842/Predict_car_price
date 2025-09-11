# Use Python 3.10.9 base image
FROM python:3.10.9-slim

# Set working directory
WORKDIR /root/code

# Install system dependencies (if needed for dash/plotly)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Expose Dash default port
EXPOSE 8050

# Run the app
# CMD ["python", "/root/code/Code/index.py"]
