import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if request is from API (JSON)
        if request.is_json:
            data = request.json
            if "features" not in data:
                return jsonify({"error": "Missing 'features' key in request"}), 400

            # Convert JSON data to numpy array
            features = np.array(data["features"]).reshape(1, -1)

        # Otherwise, check for form input (Web UI)
        else:
            # Get all form values except "wine_type"
            form_values = list(request.form.values())

            # Extract "wine_type" separately
            wine_type = int(form_values[-1])  # Last value is the wine type (0 for Red, 1 for White)

            # Convert remaining form inputs to floats
            features = [float(x) for x in form_values[:-1]]  # Exclude wine_type from numeric features

            # Convert to NumPy array
            features = np.array(features).reshape(1, -1)

        # 🚨 Add "wine_type" to features to ensure correct number of inputs (12 features)
        features_with_wine_type = np.hstack((features, [[wine_type]]))  # Append wine_type

        # Scale input data
        features_scaled = scaler.transform(features_with_wine_type)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Return JSON if API request, else render HTML
        if request.is_json:
            return jsonify({"predicted_quality": prediction})
        else:
            return render_template("index.html", prediction_text=f"Predicted Wine Quality: {prediction:.2f}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
