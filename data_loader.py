import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import numpy as np

tickers = ["JPM", "GS", "BAC", "C", "MS", "WFC", "USB", "PNC", "BK", "STT"]

# data in DataFrame laden
def load_data(tickers):
    data = yf.download(tickers, start="2020-01-01", end=datetime.now())
    return data

data = load_data(tickers)
# nur "close" column auswählen
data_close = data["Close"]

#print(data_close.isnull().sum())
data_close.dropna(inplace=True)
print(data_close.head())
print(data_close.shape)

#Kurse mit matplotlib plotten mit Legende
'''
plt.figure(figsize=(12, 6))
plt.plot(data_close)
plt.title("Kurse der Aktien")
plt.xlabel("Zeit")
plt.ylabel("Kurs")
plt.legend(tickers)
plt.show()
'''

pairs = list(combinations(tickers, 2))
results_dict = {}
for pair in pairs:
    #nur p-Wert ausgeben
    results_dict[pair] = coint(data_close[pair[0]], data_close[pair[1]])[1]

#results in DataFrame
results = pd.DataFrame(results_dict.items(), columns=["Pair", "p-Wert"])
results.sort_values(by="p-Wert", inplace=True)
#nur p < 0.05 ausgeben und nur top 5
results_significant = results[results["p-Wert"] < 0.05]
print(results_significant.head(5))

#nachfolgende Analyse für das beste Paar
best_pairs = results_significant.iloc[0]["Pair"]
ticker_a = best_pairs[0]
ticker_b = best_pairs[1]

price_a = data_close[ticker_a]
price_b = data_close[ticker_b]

#OLS Regression
X = sm.add_constant(price_a)
model = sm.OLS(price_b, X)
results = model.fit()
print(results.summary())
beta = results.params.iloc[1]
spread = price_b - beta * price_a

#Stationäritätsprüfung
stationary_check = adfuller(spread)[1]
if stationary_check < 0.05:
    print(f"Spread ist stationär mit p-Wert {stationary_check}")
else:
    print(f"Spread ist nicht stationär mit p-Wert {stationary_check}")
'''
plt.axhline(spread.mean(), color='red', linestyle='--')
plt.plot(spread)
plt.title(f"Spread von {ticker_a} - beta * {ticker_b}")
plt.xlabel("Zeit")
plt.ylabel("Spread")
plt.show()
'''

spread_lag = spread.iloc[:-1]
spread_now = spread.iloc[1:]

X_lag = sm.add_constant(spread_lag.values)
model = sm.OLS(spread_now.values, X_lag)
results = model.fit()

a = results.params[0]
b = results.params[1]

theta = -np.log(b)
mu = a / (1 - b)
sigma = np.std(results.resid)
half_life = np.log(2) / theta

print(f"θ: {theta:.4f}, μ: {mu:.4f}, σ: {sigma:.4f}, Half-Life: {half_life:.4f}, Spread-Mittelwert: {spread.mean():.4f}")

#z-score mit gesamt-Mittelwert berechnen
window = 30
rolling_mean = spread.rolling(window).mean()
rolling_std = spread.rolling(window).std()
z_score = (spread - rolling_mean) / rolling_std

signal = pd.Series(0, index=spread.index)
signal[np.abs(z_score) > 2] = 1 #long
signal[np.abs(z_score) < -2] = -1 #short

'''
plt.figure(figsize=(14,8))
# Oben: Spread mit rollierendem Mittelwert
plt.subplot(2, 1, 1)
plt.plot(spread, label="Spread")
plt.plot(rolling_mean, color="red", label="Rolling Mean")
plt.title("Spread mit rollierendem Mittelwert")
plt.legend()

# Unten: Z-Score mit Schwellenwerten
plt.subplot(2, 1, 2)
plt.plot(z_score, label="Z-Score")
plt.axhline(2.0, color="red", linestyle="--", label="Entry Short")
plt.axhline(-2.0, color="green", linestyle="--", label="Entry Long")
plt.axhline(0.5, color="orange", linestyle="--", label="Exit")
plt.axhline(-0.5, color="orange", linestyle="--")
plt.title("Z-Score mit Schwellenwerten")
plt.legend()

plt.tight_layout()
plt.show()
'''

split = len(spread) // 2
spread_test = spread.iloc[split:]
z_score_test = z_score.iloc[split:]

position = 0
entry_price = 0
pnl_list = []
trades = []

for j in range(1, len(spread_test)):
    spread_today = spread_test.iloc[j]
    spread_yesterday = spread_test.iloc[j - 1]
    daily_pnl = position * (spread_today - spread_yesterday)
    pnl_list.append(daily_pnl)
    z = z_score_test.iloc[j]
    if position == 0:
        if z > 2.0:
            position = -1                # Short: Spread ist zu hoch
            entry_price = spread_today
        elif z < -2.0:
            position = 1                 # Long: Spread ist zu niedrig
            entry_price = spread_today
    else:
        if abs(z) < 0.5:                # Spread normalisiert → Exit
            trade_pnl = position * (spread_today  - entry_price)
            trades.append(trade_pnl)
            position = 0
            entry_price = 0
        elif abs(z) > 3.5:              # Stop-Loss
            trade_pnl = position * (spread_today - entry_price)
            trades.append(trade_pnl)
            position = 0
            entry_price = 0
        

equity = np.cumsum(pnl_list)

plt.figure(figsize=(12, 6))
plt.plot(equity)
plt.title("Equity-Kurve (Out-of-Sample)")
plt.xlabel("Handelstage")
plt.ylabel("Kumulierter P&L")
plt.axhline(0, color="red", linestyle="--")
plt.show()

pnl_array = np.array(pnl_list)
sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)
equity = np.cumsum(pnl_array)
max_equity = np.maximum.accumulate(equity)
drawdown = max_equity - equity
max_drawdown = drawdown.max()
trades_array = np.array(trades)
wins = np.sum(trades_array > 0)
win_rate = wins / len(trades)

print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"Max Drawdown:     {max_drawdown:.2f}")
print(f"Anzahl Trades:    {len(trades)}")
print(f"Win Rate:         {win_rate:.1%}")
print(f"Total P&L:        {equity[-1]:.2f}")
