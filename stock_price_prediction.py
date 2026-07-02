import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42


def build_sample_stock_data(periods=120):
    """Create a small synthetic OHLCV dataset with realistic price relationships."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="B")

    close = 125 + np.cumsum(rng.normal(loc=0.12, scale=1.6, size=periods))
    open_price = close + rng.normal(loc=0, scale=1.2, size=periods)
    high = np.maximum(open_price, close) + rng.uniform(0.3, 2.5, periods)
    low = np.minimum(open_price, close) - rng.uniform(0.3, 2.5, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def prepare_features(df):
    df = df.sort_values("Date").copy()
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["Target_Close"] = df["Close"].shift(-1)
    df = df.dropna()

    features = ["Open", "High", "Low", "Volume", "Daily_Return", "MA_5"]
    return df[features], df["Target_Close"]


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    return y_val, predictions, rmse


def plot_predictions(y_val, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_val.values, label="Actual Close Price")
    plt.plot(predictions, label="Predicted Close Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Validation Day")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(show_plot=True):
    df = build_sample_stock_data()
    X, y = prepare_features(df)
    y_val, predictions, rmse = train_model(X, y)

    print(f"Validation RMSE: {rmse:.2f}")

    if show_plot:
        plot_predictions(y_val, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a simple next-day stock close predictor."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the model without opening the matplotlib chart.",
    )
    args = parser.parse_args()
    main(show_plot=not args.no_plot)
