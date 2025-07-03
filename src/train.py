import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data = pd.read_csv('data/all_stocks_5yr.csv', index_col='date', parse_dates=True)

data['Return'] = data['close'].pct_change()

data['Target'] = data['Return'].shift(-1)

data.dropna(inplace=True)

data['SMA_5'] = data['close'].rolling(5).mean()
data['SMA_20'] = data['close'].rolling(20).mean()
data['RSI'] = compute_rsi(data['close'], 14)
data['Momentum'] = data['close'] - data['close'].shift(5)

data['Volume_MA_5'] = data['volume'].rolling(5).mean()
data['Volume_Change'] = data['volume'].pct_change()

data['Daily_Range'] = (data['high'] - data['low']) / data['open']
data['Close_Open_Ratio'] = data['close'] / data['open']

data.dropna(inplace=True)

features = ['Return', 'SMA_5', 'SMA_20', 'RSI', 'Momentum', 
            'Volume_MA_5', 'Volume_Change', 'Daily_Range', 'Close_Open_Ratio']

X = data[features]
Y = data['Target']

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

print("Training Performance:")
print(f"R²: {r2_score(Y_train, train_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_train, train_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_train, train_pred):.6f}")

plt.figure(figsize=(14, 6))

print("\nTest Performance:")
print(f"R²: {r2_score(Y_test, test_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_test, test_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_test, test_pred):.6f}")  # New RMSE line

plt.figure(figsize=(14,7))
plt.plot(data.index[split_idx:], Y_test, label='Actual Returns')
plt.plot(data.index[split_idx:], test_pred, label='Predicted Returns', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Returns')
plt.show()
