# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# (Adding build-essential for XGBoost and Scikit-Learn C-extensions)
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the Hydra Engine into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run FastAPI using Uvicorn when the container launches
CMD ["uvicorn", "api.py", "--host", "0.0.0.0", "--port", "8000"]