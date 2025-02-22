import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]  # Example input
}

response = requests.post(url, json=data)
print(response.json())  # Expected output: {'predicted_quality': [5.4]}
