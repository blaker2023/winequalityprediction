import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    """Render the main HTML page for user input."""
    return render_template("index.html")  # Ensure 'index.html' is in the 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        features = [
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulfates"]),
            float(request.form["alcohol"]),
            1 if request.form["wine_type"] == "White Wine" else 0  # Convert wine type to numerical
        ]

        # Convert to NumPy array
        input_data = np.array([features])

        # Apply StandardScaler before prediction
        input_data_scaled = scaler.transform(input_data)

        # Get prediction
        prediction = model.predict(input_data_scaled)

        return render_template("index.html", prediction=f"Predicted Wine Quality: {prediction[0]:.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
