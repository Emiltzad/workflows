import datetime
import json
import os
import pytz
import yfinance as yf

# ---------------------------------------------------------------------
# 1. הגדרות ונכסים במעקב
# ---------------------------------------------------------------------
TICKERS = [
    "OPEN",
    "BBAI",
    "PLTR",
    "SOFI",
    "ICCM",
    "QS",
    "SMR",
    "NVDA",
    "AMD",
    "META",
    "GOOG",
    "MRVL",
    "ORCL",
    "SEDG", "NOW", "SKM", "PENG"
]

DATA_FILE = "stock_history_db.json"

def calculate_vwap_and_indicators(df):
    """חישוב VWAP, RSI ורצועות בולינגר מתוך נרות תוך-יומיים"""
    if len(df) < 20:
        return None
    
    last_date = df.index.date[-1]
    session_df = df[df.index.date == last_date].copy()
    
    close_ser = session_df["Close"].squeeze()
    high_ser = session_df["High"].squeeze()
    low_ser = session_df["Low"].squeeze()
    vol_ser = session_df["Volume"].squeeze()

    # VWAP
    tp = (high_ser + low_ser + close_ser) / 3
    cum_tp_vol = (tp * vol_ser).cumsum()
    cum_vol = vol_ser.cumsum()
    session_df["VWAP"] = cum_tp_vol / cum_vol

    # Bollinger Bands
    ma20 = close_ser.rolling(window=20).mean()
    std20 = close_ser.rolling(window=20).std()
    bb_upper = ma20 + (2 * std20)
    bb_lower = ma20 - (2 * std20)

    # RSI
    delta = close_ser.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    last_idx = -1
    return {
        "Close": round(float(close_ser.iloc[last_idx]), 2),
        "VWAP": round(float(session_df["VWAP"].iloc[last_idx]), 2),
        "RSI": round(float(rsi.iloc[last_idx]), 1) if not rsi.empty else "N/A",
        "BB_Upper": round(float(bb_upper.iloc[last_idx]), 2) if not bb_upper.empty else "N/A",
        "BB_Lower": round(float(bb_lower.iloc[last_idx]), 2) if not bb_lower.empty else "N/A",
        "Volume": int(vol_ser.iloc[last_idx])
    }

def fetch_daily_snapshot():
    tz_ny = pytz.timezone("America/New_York")
    now_ny = datetime.datetime.now(tz_ny)
    date_str = now_ny.strftime("%Y-%m-%d")
    
    snapshot = {
        "timestamp": now_ny.strftime("%Y-%m-%d %H:%M:%S EST"),
        "date": date_str,
        "stocks": {}
    }

    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)
            # משיכת נתונים תוך-יומיים למדדים
            intra = t.history(period="3d", interval="5m")
            indicators = calculate_metrics_safe(intra)
            
            # משיכת נתונים פונדמנטליים
            info = t.info
            
            snapshot["stocks"][ticker] = {
                "Price": indicators["Close"] if indicators else info.get("regularMarketPrice"),
                "VWAP": indicators["VWAP"] if indicators else "N/A",
                "RSI": indicators["RSI"] if indicators else "N/A",
                "Volume": indicators["Volume"] if indicators else info.get("volume"),
                "MarketCap": info.get("marketCap"),
                "ShortFloat": f"{round(info.get('shortPercentOfFloat', 0) * 100, 2)}%" if info.get('shortPercentOfFloat') else "N/A"
            }
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return snapshot

def calculate_metrics_safe(df):
    try:
        return calculate_vwap_and_indicators(df)
    except:
        return None

def update_cloud_database():
    new_snapshot = fetch_daily_snapshot()
    
    # טעינת בסיס הנתונים הקיים
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except:
                db = []
    else:
        db = []

    # הסרת כפילויות לפי תאריך במידה והורץ פעמיים באותו יום
    db = [entry for entry in db if entry.get("date") != new_snapshot["date"]]
    db.append(new_snapshot)

    # שמירה חזרה לקובץ הענן
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
        
    print(f"[{new_snapshot['timestamp']}] Database updated successfully with {len(new_snapshot['stocks'])} stocks.")

if __name__ == "__main__":
    update_cloud_database()