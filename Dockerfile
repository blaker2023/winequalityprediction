# Use an official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY train_model.py train_model.py
COPY app.py app.py
COPY wine_quality_model.pkl wine_quality_model.pkl

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the API
CMD ["python", "app.py"]