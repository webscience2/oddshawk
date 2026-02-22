FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY setup.sh ./
COPY templates/ templates/

# Data volume mount point
VOLUME ["/app/data"]

# Default: run the scanner
CMD ["python3", "main.py"]
