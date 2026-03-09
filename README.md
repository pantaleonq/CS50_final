# Pairs Trading with Ornstein-Uhlenbeck Process

## Description

This project implements a statistical arbitrage strategy (Pairs Trading) for the US banking sector. The core idea: two stocks that historically move together (are cointegrated) are identified. When the price difference (spread) between them becomes unusually large, the strategy bets on it reverting to the mean.

The project uses the Ornstein-Uhlenbeck process — a stochastic model from quantitative finance — to mathematically describe the mean-reversion dynamics of the spread and estimate the speed at which it returns to equilibrium.

## Features

- **Cointegration Screening:** Automatically tests all possible stock pairs for statistical cointegration (Engle-Granger test)
- **Spread Modeling:** Calculates the hedge ratio via OLS regression and verifies stationarity (ADF test)
- **OU Parameter Estimation:** Estimates mean-reversion speed (θ), equilibrium level (μ), and volatility (σ) through a discretized AR(1) regression
- **Z-Score Signals:** Rolling standardization of the spread with entry (|Z|>2), exit (|Z|<0.5), and stop-loss signals (|Z|>3.5)
- **Backtesting Engine:** Out-of-sample strategy test with daily P&L calculation and performance metrics
- **Web App:** Interactive Flask application with Plotly charts for visualizing spread, Z-score, and equity curve

## Project Structure

```
pairs-trading/
├── app.py                 # Flask web app
├── trading_engine.py      # Backend: all calculation functions
├── data_loader.py         # Original data script
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html         # Home page with pairs table
│   └── pair.html          # Detail page with charts and metrics
└── README.md
```

## Installation

```bash
git clone [REPO-URL]
cd pairs-trading
python -m venv venv
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open in browser: `http://127.0.0.1:5001`

## Technologies Used

- **Python** with NumPy, Pandas, SciPy, Statsmodels
- **yfinance** for historical price data from Yahoo Finance
- **Flask** as web framework
- **Plotly.js** for interactive browser-based charts

## Methodology

### 1. Cointegration

Two time series are cointegrated if a linear combination of them is stationary — meaning it fluctuates around a constant mean. The Engle-Granger test checks this for all stock pairs. Pairs with p-value < 0.05 are considered significant.

### 2. Ornstein-Uhlenbeck Process

The spread is modeled as an OU process:

```
dS = θ(μ - S)dt + σdW
```

- **θ** = Mean-reversion speed (how fast the spread returns)
- **μ** = Equilibrium level (where it returns to)
- **σ** = Spread volatility
- **Half-Life** = ln(2)/θ (days until half-way return to the mean)

Parameters are estimated via an AR(1) regression of the spread on its own lag.

### 3. Trading Signals

The spread is standardized as a rolling Z-score (30-day window):

- **Entry Short:** Z > 2.0 (spread is unusually high → expected to fall)
- **Entry Long:** Z < -2.0 (spread is unusually low → expected to rise)
- **Exit:** |Z| < 0.5 (spread has normalized)
- **Stop-Loss:** |Z| > 3.5 (spread continues to diverge)

### 4. Backtesting

The data is split in half. The first half is used for parameter estimation (in-sample), the second half for strategy testing (out-of-sample). Metrics:

- **Sharpe Ratio:** Risk-adjusted return (annualized)
- **Max Drawdown:** Largest drop from peak equity
- **Win Rate:** Proportion of profitable trades

## Results (Example: MS / STT)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.76 |
| Win Rate | 85.0% |
| Max Drawdown | 7.67 |
| Number of Trades | 20 |
| Total P&L | 26.74 |
| Half-Life | 28.1 days |

## Limitations

- **Past ≠ Future:** Cointegration can break — the statistical relationship between two stocks is not guaranteed to remain stable.
- **No real transaction costs:** The model does not account for bid-ask spreads, margin costs, or slippage.
- **Daily close prices:** In practice, pairs trades are executed intraday. Daily data may delay signals.
- **Single sector only:** The screening is limited to US banks. Other sectors may yield better or worse pairs.

## Author

pantaleonq

CS50 Final Project 2026
