import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import warnings

def calculate_rsi(close, window=20):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    bb_high = rolling_mean + (rolling_std * num_std)
    bb_low = rolling_mean - (rolling_std * num_std)
    bb_mid = rolling_mean
    
    return bb_low, bb_mid, bb_high

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_macd(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

data = pd.read_csv('data/all_stocks_5yr.csv', index_col='date', parse_dates=True)

if 'Name' in data.columns:
    stock_names = data['Name'].unique()
    
    # # Filter to one stock for simplicity
    # selected_stock = stock_names[0]
    # print(f"Using stock: {selected_stock}")
    # data = data[data['Name'] == selected_stock].copy()


data['garman_klass_vol'] = ((np.log(data['high']) - np.log(data['low'])) ** 2) / 2  - (2 * np.log(2) - 1) * (np.log(data['close']) - np.log(data['open'])) ** 2

data['rsi'] = calculate_rsi(data['close'], 20)

bb_low, bb_mid, bb_high = calculate_bollinger_bands(np.log1p(data['close']), 20)
data['bb_low'] = bb_low
data['bb_mid'] = bb_mid
data['bb_high'] = bb_high

atr = calculate_atr(data['high'], data['low'], data['close'], 14)
data['atr'] = atr.sub(atr.mean()).div(atr.std())

macd = calculate_macd(data['close'], 12, 26)
data['macd'] = macd.sub(macd.mean()).div(macd.std())

data['dollar_volume'] = (data['close']) * data['volume'] /1e6
data.dropna(inplace=True)

features = ['garman_klass_vol', 'rsi', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume']
data['Return'] = data['close'].pct_change()
data['Target'] = data['Return'].shift(-1) 

X = data[features]
Y = data['Target']

# Remove any remaining NaN values
mask = ~(X.isna().any(axis=1) | Y.isna())
X = X[mask]
Y = Y[mask]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(X_scaled) *0.8)

X_train = X_scaled[:split_idx]
Y_train = Y[:split_idx]

X_test = X_scaled[split_idx:]
Y_test = Y[split_idx:]

model = XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 5,
    subsample = 0.9,
    colsample_bytree = 0.7,
    random_state = 42,
    reg_alpha=0.1
)

model.fit(X_train, Y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nTraining Performance:")
print(f"R²: {r2_score(Y_train, train_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_train, train_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_train, train_pred):.6f}")

print("\nTest Performance:")
print(f"R²: {r2_score(Y_test, test_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_test, test_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_test, test_pred):.6f}") 

# Uncomment to plot results
# plt.figure(figsize=(14,7))
# plt.plot(Y_test.index, Y_test, label='Actual Returns')
# plt.plot(Y_test.index, test_pred, label='Predicted Returns', alpha=0.7)
# plt.legend()
# plt.title('Actual vs Predicted Returns')
# plt.show()