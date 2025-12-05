FROM jupyter/pyspark-notebook:latest

# Set working directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt .
COPY . /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8888 8501 4040

# Default command
CMD ["python", "main.py"]
