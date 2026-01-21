import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Download gold price data
data = yf.download("GC=F", start="2015-01-01")

# 2. Feature engineering
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data["Lag1"] = data["Close"].shift(1)

# 3. Create future target (next-day price)
data["Target"] = data["Close"].shift(-1)

# 4. Drop rows with NaN values ONCE
data.dropna(inplace=True)

# 5. Prepare features and target
X = data[["Open", "High", "Low", "Volume", "MA5", "MA10", "Lag1"]]
y = data["Target"]

# 6. Train-test split (time-series safe)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 7. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predict
predictions = model.predict(X_test)

# 9. Evaluate
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 10. Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("Gold Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
