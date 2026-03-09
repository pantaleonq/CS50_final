from flask import Flask, render_template
from trading_engine import load_data, find_pairs, calculate_spread, estimate_ou, calculate_zscore, backtest, TICKER_NAMES
import json

app = Flask(__name__)

tickers = ["JPM", "GS", "BAC", "C", "MS", "WFC", "USB", "PNC", "BK", "STT"]
data_close = load_data(tickers)
pairs = find_pairs(data_close)

@app.route("/")
def index():
    pairs_list = []
    for _, row in pairs.iterrows():
        a = row["Pair"][0]
        b = row["Pair"][1]
        pairs_list.append({
            "pair": f"{TICKER_NAMES[a]} / {TICKER_NAMES[b]}",
            "ticker_a": a,
            "ticker_b": b,
            "p_value": round(row["p-Wert"], 4)
        })
    return render_template("index.html", pairs=pairs_list)

@app.route("/pair/<ticker_a>/<ticker_b>")
def pair_detail(ticker_a, ticker_b):
    beta, spread = calculate_spread(data_close, ticker_a, ticker_b)
    theta, mu, sigma, half_life = estimate_ou(spread)
    z_score = calculate_zscore(spread, 30)
    equity, n_trades, sharpe, max_dd, win_rate = backtest(spread, z_score)

    # Daten für die Charts als JSON vorbereiten
    spread_data = json.dumps(spread.values.tolist())
    z_score_data = json.dumps(z_score.dropna().values.tolist())
    equity_data = json.dumps(equity.tolist())

    return render_template("pair.html",
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        beta=round(beta, 4),
        theta=round(theta, 4),
        mu=round(mu, 4),
        half_life=round(half_life, 1),
        sharpe=round(sharpe, 2),
        max_dd=round(max_dd, 2),
        n_trades=n_trades,
        win_rate=round(win_rate * 100, 1),
        total_pnl=round(equity[-1], 2),
        spread_data=spread_data,
        z_score_data=z_score_data,
        equity_data=equity_data
    )

if __name__ == "__main__":
    print(app.url_map)
    app.run(debug=True, port=5001)