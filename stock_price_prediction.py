import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)

data = {
    "Date": pd.date_range(start="2023-01-01", periods=120, freq="D"),
    "Open": np.random.uniform(100, 150, 120),
    "High": np.random.uniform(150, 170, 120),
    "Low": np.random.uniform(90, 120, 120),
    "Close": np.random.uniform(100, 160, 120),
    "Volume": np.random.randint(1_000_000, 5_000_000, 120)
}

df = pd.DataFrame(data)
df = df.sort_values("Date")
df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["Target_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)


features = ["Open", "High", "Low", "Volume", "Daily_Return", "MA_5"]
X = df[features]
y = df["Target_Close"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")


plt.figure(figsize=(10, 5))
plt.plot(y_val.values, label="Actual Close Price")
plt.plot(y_pred, label="Predicted Close Price")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()