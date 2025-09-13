# Base image with Python and Uvicorn
FROM tiangolo/uvicorn-gunicorn:python3.9

# Set up the working directory
WORKDIR /backend

# Copy the backend code to the container
COPY . /backend

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
