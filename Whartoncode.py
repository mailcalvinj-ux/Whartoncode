import pandas as pd
import numpy as np
import yfinance as yf

DEFAULT_ESG_SCORE = 50   # fallback value if ESG not provided

# --------------------------
# Scoring functions
# --------------------------
def score_pe(pe):
    if pe < 10: return 1.0
    elif pe < 20: return 0.8
    elif pe < 40: return 0.5
    else: return 0.2

def score_roe(roe):
    if roe > 0.30: return 1.0
    elif roe > 0.20: return 0.8
    elif roe > 0.10: return 0.6
    elif roe > 0.05: return 0.4
    else: return 0.2

def score_volatility(vol):
    if vol < 0.15: return 1.0
    elif vol < 0.25: return 0.8
    elif vol < 0.35: return 0.5
    else: return 0.3

def score_dividend(dy):
    if dy > 0.04: return 1.0
    elif dy > 0.02: return 0.8
    elif dy > 0.01: return 0.5
    else: return 0.3

# --------------------------
# Manual ESG input
# --------------------------
def get_manual_esg_score(ticker, esg_dict):
    """Return ESG score from user dictionary, fallback to default."""
    value = esg_dict.get(ticker.upper(), None)
    return DEFAULT_ESG_SCORE if value is None else value

# --------------------------
# Fetch stock fundamentals
# --------------------------
def get_stock_data(tickers, esg_dict=None):
    if esg_dict is None:
        esg_dict = {}

    data = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe_ratio       = info.get("trailingPE", np.nan)
        roe            = info.get("returnOnEquity", np.nan)
        volatility     = info.get("beta", np.nan)
        dividend_yield = info.get("dividendYield", np.nan)

        # Replace with manual ESG lookup
        esg_score = get_manual_esg_score(ticker, esg_dict)

        # Fallback defaults
        if np.isnan(pe_ratio): pe_ratio = 25
        if np.isnan(roe): roe = 0.10
        if np.isnan(volatility): volatility = 0.25
        if np.isnan(dividend_yield): dividend_yield = 0.01

        data.append({
            "ticker": ticker,
            "pe_ratio": pe_ratio,
            "roe": roe,
            "volatility": volatility,
            "dividend_yield": dividend_yield,
            "esg_score": esg_score
        })
    
    return pd.DataFrame(data)

# --------------------------
# Compute SIR-JVP score
# --------------------------
def sir_jvp_absolute(df, weights=None):
    if weights is None:
        weights = {
            "pe": 0.2,
            "roe": 0.25,
            "volatility": 0.25,
            "dividend": 0.15,
            "esg": 0.15
        }

    scores = []
    for _, row in df.iterrows():
        pe_score = score_pe(row["pe_ratio"])
        roe_score = score_roe(row["roe"])
        vol_score = score_volatility(row["volatility"])
        div_score = score_dividend(row["dividend_yield"])
        esg_score = row["esg_score"] / 100

        total = (
            weights["pe"] * pe_score +
            weights["roe"] * roe_score +
            weights["volatility"] * vol_score +
            weights["dividend"] * div_score +
            weights["esg"] * esg_score
        )

        scores.append(total)

    df["sir_jvp_score"] = scores
    return df.sort_values("sir_jvp_score", ascending=False)

# --------------------------
# TICKERS + MANUAL ESG SCORES (Inserted from your images)
# --------------------------
tickers = [
    "XOM", "BP", "SHEL", "BA", "MCG", "GLEN", "CURY.L", "MPC",
    "ALB", "AMAT", "BCDRF", "BEP", "DPZ", "GLD", "KO", "LLY",
    "MOAT", "MSFT", "NEE", "NVDA", "OPEN", "SLV", "XLV", "XYL"
]

manual_esg_scores = {
    "XOM": 36,
    "BP": 38,
    "SHEL": 41,
    "BA": 40,
    "MCG": 42,
    "GLEN": 19,
    "CURY.L": 45,
    "MPC": 51,
    "ALB": 61,
    "AMAT": 43,
    "BCDRF": 57,
    "BEP": 57,
    "DPZ": 23,
    "GLD": None,
    "KO": 42,
    "LLY": 40,
    "MOAT": None,
    "MSFT": 51,
    "NEE": 36,
    "NVDA": 61,
    "OPEN": None,
    "SLV": None,
    "XLV": None,
    "XYL": 46
}

# --------------------------
# RUN
# --------------------------
stock_data = get_stock_data(tickers, manual_esg_scores)
result_df = sir_jvp_absolute(stock_data)

print(result_df[["ticker", "sir_jvp_score"]])
