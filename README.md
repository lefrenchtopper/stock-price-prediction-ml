# Stock Price Prediction using Machine Learning
## Problem Statement

The goal of this project is to predict the **next day’s closing stock price** using historical stock market data. This is a regression problem and demonstrates a complete beginner-friendly machine learning workflow suitable for an ML internship.

## Dataset Description

The dataset simulates realistic stock market data with the following features:

* **Date** – Trading date
* **Open** – Opening price of the stock
* **High** – Highest price of the day
* **Low** – Lowest price of the day
* **Close** – Closing price of the stock
* **Volume** – Number of shares traded

The target variable is the **next day’s closing price**.

## Approach

1. Loaded structured financial data using Pandas
2. Sorted data chronologically to avoid data leakage
3. Performed feature engineering:

   * **Daily Return**: Measures intraday price movement
   * **5-Day Moving Average**: Captures short-term price trends
4. Split data into training and validation sets without shuffling
5. Scaled numerical features using StandardScaler
6. Trained a **Linear Regression** model
7. Evaluated performance using **Root Mean Squared Error (RMSE)**
8. Visualized actual vs predicted stock prices


## Tools Used

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib



## Results

The Linear Regression model was able to capture basic trends in the stock price data. RMSE was used to evaluate prediction accuracy, as it penalizes large errors and is suitable for regression problems involving continuous values like prices.


## Future Improvements

* Use real stock data from Yahoo Finance or NSE
* Try more advanced models such as Random Forest
* Add technical indicators like RSI or MACD
* Perform cross-validation
* Extend prediction to multiple future days


