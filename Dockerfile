# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all necessary files
COPY requirements.txt requirements.txt
COPY train_model.py train_model.py
COPY app.py app.py
COPY model.pkl model.pkl
COPY scaler.pkl scaler.pkl

# **FIX: Copy templates directory**
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
