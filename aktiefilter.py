import yfinance as yf
import pandas as pd
import numpy as np
import time

# === 1. Hämta alla tickers från NASDAQ ===
url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
tickers_df = pd.read_csv(url, sep="|")
symbols = [s for s in tickers_df["Symbol"].dropna().tolist() if s.isalpha() and len(s) <= 5]

# === 2. Hjälpfunktion för RSI ===
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

results = []

# === 3. Analysera aktier i batchar ===
for i in range(0, len(symbols), 10):
    batch = symbols[i:i+10]
    try:
        data = yf.download(batch, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if data.empty: continue
        closes = data["Close"] if isinstance(data["Close"], pd.DataFrame) else pd.DataFrame({batch[0]: data["Close"]})

        for ticker in closes.columns:
            prices = closes[ticker].dropna()
            if len(prices) < 30:
                continue
            last = prices.iloc[-1]
            sma50 = prices.rolling(50).mean().iloc[-1]
            sma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else np.nan
            rsi = calc_rsi(prices).iloc[-1]

            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE", np.nan)
            pb = info.get("priceToBook", np.nan)
            roe = info.get("returnOnEquity", np.nan)
            debt_eq = info.get("debtToEquity", np.nan)
            profit_margin = info.get("profitMargins", np.nan)
            eps_growth = info.get("earningsQuarterlyGrowth", np.nan)
            revenue_growth = info.get("revenueGrowth", np.nan)

            score = 0
            # --- Teknisk analys ---
            if last > sma50: score += 1
            if not np.isnan(sma200) and last > sma200: score += 1
            if 40 < rsi < 70: score += 1

            # --- Fundamental analys ---
            if not np.isnan(pe):
                # Lägre P/E = bättre poäng
                if pe < 15: score += 2
                elif pe < 25: score += 1
            if not np.isnan(pb) and pb < 5: score += 1
            if not np.isnan(roe) and roe > 0.10: score += 1
            if not np.isnan(profit_margin) and profit_margin > 0.05: score += 1
            if not np.isnan(debt_eq) and debt_eq < 2: score += 1
            if not np.isnan(eps_growth) and eps_growth > 0.10: score += 1
            if not np.isnan(revenue_growth) and revenue_growth > 0.10: score += 1

            if score >= 4:
                results.append({
                    "symbol": ticker,
                    "price": round(last, 2),
                    "PE": round(pe, 2) if not np.isnan(pe) else None,
                    "PB": round(pb, 2) if not np.isnan(pb) else None,
                    "ROE": round(roe, 2) if not np.isnan(roe) else None,
                    "RSI": round(rsi, 2),
                    "ProfitMargin": round(profit_margin, 2) if not np.isnan(profit_margin) else None,
                    "EPS_Growth": round(eps_growth, 2) if not np.isnan(eps_growth) else None,
                    "RevenueGrowth": round(revenue_growth, 2) if not np.isnan(revenue_growth) else None,
                    "Score": score
                })
        print(f"✅ {i+10}/{len(symbols)} analyserade")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Batch fel: {e}")
        continue

df = pd.DataFrame(results)
df = df.sort_values("Score", ascending=False)
top = df.head(50)
top.to_csv("top_stocks.csv", index=False)

print("\n✅ Klart! Sparade de 50 bästa aktierna (alla storlekar) i top_stocks.csv")
