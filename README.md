# Stock Price Prediction using Machine Learning

This project predicts the next trading day's closing stock price from historical
OHLCV-style market data. It uses a small synthetic dataset so the full workflow
can run locally without downloading external data.

## What the script does

1. Builds a reproducible sample stock dataset with realistic price relationships
2. Sorts observations chronologically to avoid look-ahead leakage
3. Adds simple engineered features:
   - Daily return
   - 5-day moving average
4. Splits the data into training and validation sets without shuffling
5. Scales numerical features with `StandardScaler`
6. Trains a `LinearRegression` model
7. Evaluates validation performance with RMSE
8. Plots actual vs. predicted close prices

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Usage

Run the full script and show the chart:

```bash
python stock_price_prediction.py
```

Run without opening a plot window:

```bash
python stock_price_prediction.py --no-plot
```

## Tools Used

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib

## Future Improvements

- Replace the synthetic dataset with real market data from a source such as Yahoo Finance
- Add more technical indicators, such as RSI, MACD, and Bollinger Bands
- Compare baseline models against tree-based and time-series models
- Add automated tests for feature engineering and data splitting
- Add GitHub Actions to run linting and tests on pull requests
