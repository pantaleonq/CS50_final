import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import numpy as np

TICKER_NAMES = {
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "BAC": "Bank of America",
    "C": "Citigroup",
    "MS": "Morgan Stanley",
    "WFC": "Wells Fargo",
    "USB": "U.S. Bancorp",
    "PNC": "PNC Financial",
    "BK": "Bank of New York Mellon",
    "STT": "State Street"
}

def load_data(tickers):
    data = yf.download(tickers, start="2020-01-01", end=datetime.now())
    data_close = data["Close"].dropna()
    return data_close

def find_pairs(data_close):
    pairs = list(combinations(data_close.columns, 2))
    results_dict = {}
    for pair in pairs:
        #nur p-Wert ausgeben
        results_dict[pair] = coint(data_close[pair[0]], data_close[pair[1]])[1]
    #results in DataFrame
    results = pd.DataFrame(results_dict.items(), columns=["Pair", "p-Wert"])
    results.sort_values(by="p-Wert", inplace=True)
    #nur p < 0.05 ausgeben und nur top 5
    results_significant = results[results["p-Wert"] < 0.05]
    return results_significant

def calculate_spread(data_close, ticker_a, ticker_b):
    price_a = data_close[ticker_a]
    price_b = data_close[ticker_b]

    #OLS Regression
    X = sm.add_constant(price_a)
    model = sm.OLS(price_b, X)
    results = model.fit()
    beta = results.params.iloc[1]
    spread = price_b - beta * price_a
    return beta, spread

def estimate_ou(spread):
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
    return theta, mu, sigma, half_life

def calculate_zscore(spread, window):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    z_score = (spread - rolling_mean) / rolling_std
    return z_score

def backtest(spread, z_score):
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
            
    pnl_array = np.array(pnl_list)
    sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)
    equity = np.cumsum(pnl_array)
    max_equity = np.maximum.accumulate(equity)
    drawdown = max_equity - equity
    max_drawdown = drawdown.max()
    trades_array = np.array(trades)
    wins = np.sum(trades_array > 0)
    win_rate = wins / len(trades)
    return equity, len(trades), sharpe, max_drawdown, win_rate

if __name__ == "__main__":
    tickers = ["JPM", "GS", "BAC", "C", "MS", "WFC", "USB", "PNC", "BK", "STT"]
    
    data_close = load_data(tickers)
    pairs = find_pairs(data_close)
    print(pairs)
    
    best_pair = pairs.iloc[0]["Pair"]
    beta, spread = calculate_spread(data_close, best_pair[0], best_pair[1])
    
    theta, mu, sigma, half_life = estimate_ou(spread)
    print(f"Half-Life: {half_life:.1f} Tage")
    
    z_score = calculate_zscore(spread, 30)
    equity, n_trades, sharpe, max_dd, win_rate = backtest(spread, z_score)
    
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total P&L: {equity[-1]:.2f}")
