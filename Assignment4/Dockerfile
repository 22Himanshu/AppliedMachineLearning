# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy required files
COPY app.py score.py best_model.pkl /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
