import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load datasets
red_wine = pd.read_csv("winequality-red.csv", delimiter=";")
white_wine = pd.read_csv("winequality-white.csv", delimiter=";")

# Add a column to differentiate red and white wine
red_wine["wine_type"] = 0  # 0 for red wine
white_wine["wine_type"] = 1  # 1 for white wine

# Combine datasets
df = pd.concat([red_wine, white_wine], axis=0)

# Separate features (X) and target variable (y)
X = df.drop(columns=['quality'])  # Features
y = df['quality']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler together
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model and scaler saved successfully!")
