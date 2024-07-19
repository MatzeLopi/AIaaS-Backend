# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /app
COPY ./backend_api /code/app

# Copy requirements.txt into code 
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Command to run the application
CMD ["fastapi", "run","app/main.py", "--host", "0.0.0.0", "--port", "8000"]

