# Use a slim Python base image
FROM python:3.11-slim

# Install OS dependencies (soundfile + ffmpeg needs these)
RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /text-to-speech

# Copy code and requirements into container
COPY . /text-to-speech

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]
