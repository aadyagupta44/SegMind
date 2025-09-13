# Use the official Python 3.9 image as the base image
FROM python:3.9

# Make a /frontend directory inside the container
RUN mkdir /frontend

# Set the working directory inside the container to /frontend
WORKDIR /frontend

# Copy the contents of the current directory into /frontend inside the container
COPY . /frontend

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port to allow external access
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
