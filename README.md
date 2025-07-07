# 📈 Stock Return Prediction with XGBoost
A machine learning project that predicts next-day stock returns using technical indicators derived from 5 years of historical stock price data. Built with Python, this project leverages the power of XGBoost to model patterns in stock volatility, momentum, and trading volume.


# 🧠 Overview
This project uses data from the S&P 500 to train a regression model that estimates the next day's return. The goal is to explore whether technical features like RSI, Bollinger Bands, MACD, and others can help forecast short-term price movement.


# 📂 Dataset
The dataset (all_stocks_5yr.csv) contains:

5 years of daily stock prices from various companies in the S&P 500

Columns include open, high, low, close, volume, and Name (ticker symbol)

Data was preprocessed and filtered for all stocks in the S&P 500 but can be minimised to only one

# 📈 Features
The following technical indicators were calculated and used as input features:

Garman-Klass Volatility

Relative Strength Index (RSI)

Bollinger Bands

Average True Range (ATR)

MACD (standardised)

Dollar Volume (traded value in millions)

The target variable is the next-day return based on the percentage change in closing price.


# 📊 Results
The model shows moderate performance on the training data but fails to generalise well on the test set, suggesting overfitting. Results are approximately:

**Training Performance:**<br>
R²: ~0.49

MAE: ~0.011

RMSE: ~0.036

**Test Performance:**<br>
R²: ~-0.04

MAE: ~0.010

RMSE: ~0.029

These results indicate that while the model captures some signal during training, it struggles to perform reliably on unseen data. Further tuning and feature engineering are required to improve test generalisation.


# 🔧 Future Improvements
Improvements I plan to make in the future once I get better at making machine learning models:

✅ Add moving averages, momentum indicators, and market context

✅ Implement time series cross-validation

✅ Use hyperparameter optimisation (e.g. GridSearchCV, Optuna)

✅ Expand from single-stock to multi-stock predictions

✅ Integrate live data feeds from Yahoo Finance

✅ Explore classification tasks (e.g. predicting up/down instead of continuous returns)
