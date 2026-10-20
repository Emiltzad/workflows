import datetime
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    "^VIX",
    "BBAI",
    "SMR",
    "MRVL",
    "GOOG",
    "ICCM",
    "SOFI",
    "QS",
    "OPEN",
    "PLTR",
    "NVDA",
    "AMD",
    "ORCL",
    "SEDG",
    "NOW",
    "SKM",
    "PENG",
    "META",
]
DB_FILENAME = "stock_history_db.json"


def calculate_indicators(df):
  if len(df) < 34:
    return {}

  delta = df["Close"].diff()
  gain = delta.where(delta > 0, 0).rolling(14).mean()
  loss = -delta.where(delta < 0, 0).rolling(14).mean()
  rsi = round(float(100 - (100 / (1 + (gain / loss).iloc[-1]))), 1)

  vol_mean_20 = df["Volume"].rolling(20).mean().iloc[-1]
  rvol = (
      round(float(df["Volume"].iloc[-1] / vol_mean_20), 2)
      if vol_mean_20 > 0
      else 1.0
  )

  sma20 = df["Close"].rolling(20).mean()
  std20 = df["Close"].rolling(20).std()
  bb_upper = round(float(sma20.iloc[-1] + (std20.iloc[-1] * 2)), 2)
  bb_lower = round(float(sma20.iloc[-1] - (std20.iloc[-1] * 2)), 2)

  tp = (df["High"] + df["Low"] + df["Close"]) / 3
  mf = tp * df["Volume"]
  mfi_series = 100 - (
      100
      / (
          1
          + (
              mf.where(tp.diff() > 0, 0).rolling(14).sum()
              / mf.where(tp.diff() < 0, 0).rolling(14).sum()
          )
      )
  )
  mfi = round(float(mfi_series.iloc[-1]), 1) if not pd.isna(mfi_series.iloc[-1]) else "N/A"  # type: ignore

  return {"RSI": rsi, "RVOL": rvol, "BB_Up": bb_upper, "BB_Low": bb_lower, "MFI": mfi}


def fetch_and_store_data():
  now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  today_str = datetime.datetime.now().strftime("%Y-%m-%d")

  print(f"[{now_str}] מתחיל איסוף נתונים מוסדיים אוטומטי בענן...")

  stocks_data = {}

  for symbol in TICKERS:
    try:
      ticker = yf.Ticker(symbol)
      df = ticker.history(period="1y")

      if df.empty:
        print(f"⚠️ אין נתונים עבור {symbol}")
        continue

      curr = float(df["Close"].iloc[-1])
      prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr
      daily_pct = round(((curr - prev) / prev) * 100, 2)

      info = ticker.info
      cap = info.get("marketCap")
      cap_str = (
          f"{cap / 1_000_000_000:.2f}B"
          if cap and cap > 1_000_000_000
          else (f"{cap / 1_000_000:.2f}M" if cap else "N/A")
      )

      ind = calculate_indicators(df)

      stocks_data[symbol] = {
          "Price": round(curr, 2),
          "Daily %": daily_pct,
          "Market Cap": cap_str,
          "RSI": ind.get("RSI", "N/A"),
          "RVOL": ind.get("RVOL", "N/A"),
          "MFI": ind.get("MFI", "N/A"),
          "BB_Up": ind.get("BB_Up", "N/A"),
          "BB_Low": ind.get("BB_Low", "N/A"),
      }
      print(f"✅ נאסף בהצלחה: {symbol} | מחיר: ${curr}")

    except Exception as e:
      print(f"❌ שגיאה בשליפת {symbol}: {e}")

  new_snapshot = {"timestamp": now_str, "date": today_str, "stocks": stocks_data}

  history_data = []
  if os.path.exists(DB_FILENAME):
    try:
      with open(DB_FILENAME, "r", encoding="utf-8") as f:
        content = f.read()
        if content.strip():
          history_data = json.loads(content)
    except Exception as e:
      print(f"⚠️ שגיאה בקריאת היסטוריה קיימת: {e}")

  history_data.append(new_snapshot)

  try:
    with open(DB_FILENAME, "w", encoding="utf-8") as f:
      json.dump(history_data, f, indent=4, ensure_ascii=False)
    print(f"💾 קובץ מסד הנתונים {DB_FILENAME} עודכן בהצלחה בענן!")
  except Exception as e:
    print(f"❌ שגיאה בשמירת הקובץ: {e}")


if __name__ == "__main__":
  fetch_and_store_data()