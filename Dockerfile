FROM python:3.10-slim

# Copy all project files to container root
COPY . /

# Set working directory to root to make sure all relative paths resolve from /
WORKDIR /

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to launch your script

ENTRYPOINT ["python", "/BME-X.py"]
