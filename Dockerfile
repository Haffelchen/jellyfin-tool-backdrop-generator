FROM python:3.9.23-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy script and entrypoint
COPY . /app

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Add shell script to parse env vars and run Python
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the new entrypoint
ENTRYPOINT ["/entrypoint.sh"]
