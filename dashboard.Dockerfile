# Use a slim Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Copy dependencies and install
# Note: Assuming this is built from the project root.
COPY requirements-dashboard.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the Cloud Run port
EXPOSE 8080

# Run standard Streamlit or Dash app.
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
