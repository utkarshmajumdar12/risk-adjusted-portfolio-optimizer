

from datetime import datetime, timedelta
import warnings, sys, numpy as np, pandas as pd, scipy.optimize as sco, yfinance as yf
warnings.filterwarnings("ignore")                        # keep the log clean

# ─────────────────────────── USER SETTINGS ────────────────────────────── #
TICKERS        = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # any liquid symbols
YEARS_HISTORY  = 2                                          # how much history
USE_FRED_RATE  = False                                      # set True with API key
FRED_API_KEY   = "YOUR_FRED_API_KEY_HERE"                   # optional
RISK_FREE_RATE = 0.02                                       # fallback risk-free rate
MC_PORTFOLIOS  = 10_000                                     # efficient-frontier dots
# ───────────────────────────────────────────────────────────────────────── #

def fetch_prices(tickers, start, end):
    """Download daily Adjusted Close prices from Yahoo Finance."""
    raw = yf.download(tickers, start=start, end=end, progress=False)

    # Debug print to check columns
    print("Columns downloaded:", raw.columns)

    # Check for MultiIndex columns (multiple tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].dropna(how="all")
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].dropna(how="all")
            print("Using 'Close' prices instead of 'Adj Close'")
        else:
            raise ValueError("Adjusted Close or Close prices not found in downloaded data")
    else:
        # Single ticker case with normal columns
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in raw.columns:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            print("Using 'Close' prices instead of 'Adj Close'")
        else:
            raise ValueError("Adjusted Close or Close prices not found in downloaded data")

    return prices.ffill().bfill()

def get_risk_free_rate():
    """Return latest 3-month T-Bill rate from FRED or the fixed fallback."""
    if not USE_FRED_RATE:
        return RISK_FREE_RATE
    try:
        from fredapi import Fred                          # free, needs API key
        fred = Fred(api_key=FRED_API_KEY)
        rate = fred.get_series_latest_release("TB3MS").iloc[-1] / 100
        return float(rate)
    except Exception:                                     # network / auth error
        return RISK_FREE_RATE

def annualised_stats(returns):
    mu  = returns.mean() * 252
    cov = returns.cov()  * 252
    return mu, cov

def portfolio_performance(weights, mu, cov, rf):
    ret   = np.dot(weights, mu)
    vol   = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

def optimise_portfolio(mu, cov, rf, objective="sharpe"):
    n = len(mu)
    bounds     = tuple((0, 1) for _ in range(n))
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init_guess = np.repeat(1/n, n)

    if objective == "vol":
        func = lambda w: portfolio_performance(w, mu, cov, rf)[1]
    elif objective == "sharpe":
        func = lambda w: -portfolio_performance(w, mu, cov, rf)[2]
    else:
        raise ValueError("objective must be 'vol' or 'sharpe'")

    res = sco.minimize(func, init_guess, method="SLSQP",
                       bounds=bounds, constraints=constraint)
    return res.x

def monte_carlo(mu, cov, rf, trials=5_000):
    n = len(mu)
    weights, rets, vols, sharpes = [], [], [], []
    for _ in range(trials):
        w = np.random.random(n); w /= w.sum()
        r, v, s = portfolio_performance(w, mu, cov, rf)
        weights.append(w); rets.append(r); vols.append(v); sharpes.append(s)
    return np.array(weights), np.array(rets), np.array(vols), np.array(sharpes)

def pretty_weights(weights, tickers):
    return pd.DataFrame({"Asset": tickers,
                         "Weight": weights,
                         "Percent": (weights*100).round(2).astype(str)+"%"}).set_index("Asset")

def main():
    # 1) fetch data -----------------------------------------------------------------
    end   = datetime.now()
    start = end - timedelta(days=YEARS_HISTORY*365)
    prices = fetch_prices(TICKERS, start, end)
    returns = np.log(prices/prices.shift(1)).dropna()

    # 2) stats ----------------------------------------------------------------------
    rf  = get_risk_free_rate()
    mu, cov = annualised_stats(returns)

    # 3) optimise -------------------------------------------------------------------
    w_max_sharpe = optimise_portfolio(mu, cov, rf, "sharpe")
    w_min_vol    = optimise_portfolio(mu, cov, rf, "vol")
    w_equal      = np.repeat(1/len(TICKERS), len(TICKERS))

    # Store performance stats (without weights to avoid DataFrame error)
    stats = {}
    weights_dict = {}
    for label, w in [("Max Sharpe", w_max_sharpe),
                     ("Min Volatility", w_min_vol),
                     ("Equal Weight", w_equal)]:
        ret, vol, sharpe = portfolio_performance(w, mu, cov, rf)
        stats[label] = (ret, vol, sharpe)
        weights_dict[label] = w

    # 4) monte-carlo ---------------------------------------------------------------
    _, mc_ret, mc_vol, mc_sharpe = monte_carlo(mu, cov, rf, MC_PORTFOLIOS)

    # 5) present results ------------------------------------------------------------
    print("\n================ PORTFOLIO RESULTS ================")
    table = pd.DataFrame(stats, index=["Return", "Volatility", "Sharpe"]).T
    table[["Return", "Volatility"]] = (table[["Return", "Volatility"]]*100).round(2).astype(str)+" %"
    table["Sharpe"] = table["Sharpe"].round(3)
    print(table, "\n")

    print("Max Sharpe Portfolio Weights:")
    print(pretty_weights(w_max_sharpe, TICKERS), "\n")

    print("Min Volatility Portfolio Weights:")
    print(pretty_weights(w_min_vol, TICKERS), "\n")

    print("Equal Weight Portfolio Weights:")
    print(pretty_weights(w_equal, TICKERS), "\n")

    # 6) quick plot -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt, seaborn as sns
        sns.set_style("whitegrid"); plt.figure(figsize=(10,8))
        sc = plt.scatter(mc_vol, mc_ret, c=mc_sharpe, cmap="viridis", s=4, alpha=0.6)
        plt.colorbar(sc, label="Sharpe Ratio"); plt.xlabel("Annualised Volatility")
        plt.ylabel("Annualised Return"); plt.title("Efficient Frontier (Monte Carlo)")
        
        # highlight key portfolios
        for lbl, w, c in [("★ Max Sharpe", w_max_sharpe, "red"),
                          ("● Min Vol",   w_min_vol,    "orange"),
                          ("■ Equal Wt",  w_equal,      "black")]:
            r, v, _ = portfolio_performance(w, mu, cov, rf)
            plt.scatter(v, r, c=c, label=lbl, edgecolors="white", s=100, linewidth=2)
        
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped –", e)

    print("Risk-free rate used:", f"{rf*100:.2f}%")
    print("Data period:", f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    sys.exit(main())
