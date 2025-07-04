# 📈 Stock Return Prediction with XGBoost
A machine learning project that predicts next-day stock returns using technical indicators derived from 5 years of historical stock price data. Built with Python, this project leverages the power of XGBoost to model patterns in stock volatility, momentum, and trading volume.


# 🧠 Overview
This project uses data from multiple U.S. stocks to train a regression model that estimates the next day's return. The goal is to explore whether technical features like RSI, Bollinger Bands, MACD, and others can help forecast short-term price movement.


# 📂 Dataset
The dataset (all_stocks_5yr.csv) contains:

5 years of daily stock prices from various companies

Columns include open, high, low, close, volume, and Name (ticker symbol)

Data was preprocessed and filtered for one selected stock (can be expanded to all)

# 📈 Features
The following technical indicators were calculated and used as input features:

Garman-Klass Volatility

Relative Strength Index (RSI)

Bollinger Bands (log-scaled)

Average True Range (ATR) (standardized)

MACD (standardized)

Dollar Volume (traded value in millions)

The target variable is the next-day return based on the percentage change in closing price.


# 📊 Results
The model shows moderate performance on the training data but fails to generalize well on the test set, suggesting overfitting. Results are approximately:

Training Performance:
R²: ~0.49

MAE: ~0.011

RMSE: ~0.036

Test Performance:
R²: ~-0.04

MAE: ~0.010

RMSE: ~0.029

These results indicate that while the model captures some signal during training, it struggles to perform reliably on unseen data. Further tuning and feature engineering are required to improve test generalization.


# 🔧 Future Improvements
✅ Add moving averages, momentum indicators, and market context

✅ Implement time series cross-validation

✅ Use hyperparameter optimization (e.g. GridSearchCV, Optuna)

✅ Expand from single-stock to multi-stock predictions

✅ Integrate live data feeds from Yahoo Finance

✅ Explore classification tasks (e.g. predicting up/down instead of continuous returns)
