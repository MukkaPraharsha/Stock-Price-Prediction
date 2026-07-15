# ==========================================================
# Stock Price Prediction Using Machine Learning
# Algorithm: Linear Regression
# ==========================================================

# -------------------------
# Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("stock.csv/stock.csv/stocks/A.csv")

print("First 5 Rows")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

# -------------------------
# Data Cleaning
# -------------------------
data = data.dropna()

# If Date column exists, sort by date
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

print("\nData cleaned successfully.")

# -------------------------
# Feature Selection
# -------------------------
X = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# -------------------------
# Model Training
# -------------------------
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# -------------------------
# Prediction
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error :", round(mae,2))
print("Mean Squared Error  :", round(mse,2))
print("Root Mean Square Error :", round(rmse,2))
print("R2 Score :", round(r2,4))

# -------------------------
# Actual vs Predicted Table
# -------------------------
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nFirst 10 Predictions")
print(result.head(10))

# -------------------------
# Save Predictions
# -------------------------
result.to_csv("Predicted_Stock_Prices.csv", index=False)

print("\nPrediction file saved successfully.")

# -------------------------
# Graph 1
# -------------------------
plt.figure(figsize=(12,6))

plt.plot(
    result["Actual Price"].values,
    color="blue",
    label="Actual Price"
)

plt.plot(
    result["Predicted Price"].values,
    color="red",
    label="Predicted Price"
)

plt.title("Stock Price Prediction")

plt.xlabel("Test Samples")

plt.ylabel("Closing Price")

plt.legend()

plt.grid(True)

plt.show()

# -------------------------
# Graph 2
# -------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.title("Actual vs Predicted Prices")

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.grid(True)

plt.show()

# -------------------------
# Predict Next Closing Price
# -------------------------
last = data.iloc[-1]

next_input = pd.DataFrame({
    "Open":[last["Open"]],
    "High":[last["High"]],
    "Low":[last["Low"]],
    "Volume":[last["Volume"]]
})

next_price = model.predict(next_input)

print("\n======================================")
print("Predicted Next Day Closing Price")
print("₹", round(next_price[0],2))
print("======================================")

# -------------------------
# Conclusion
# -------------------------
print("""
Project Completed Successfully

Project Title:
Stock Price Prediction Using Machine Learning

Algorithm Used:
Linear Regression

Features Used:
Open
High
Low
Volume

Target:
Close Price

Outputs:
1. Actual vs Predicted Table
2. Evaluation Metrics
3. Line Graph
4. Scatter Plot
5. Next Day Closing Price Prediction
""")