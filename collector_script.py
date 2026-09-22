import datetime
import json
import os
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# רשימת המניות למעקב
SYMBOLS = [
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

DB_FILE = "stock_history_db.json"


def get_fundamentals(symbol):
    if symbol.startswith("^"):
        return "N/A", "N/A", "N/A", "0.0%", "N/A"
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        cap = info.get("marketCap")
        cap_str = (
            f"{cap / 1_000_000_000:.2f}B"
            if cap and cap > 1_000_000_000
            else (f"{cap / 1_000_000:.2f}M" if cap else "N/A")
        )
        pe = (
            round(info.get("trailingPE"), 2)
            if info.get("trailingPE")
            else "N/A"
        )
        pb = (
            round(info.get("priceToBook"), 2)
            if info.get("priceToBook")
            else "N/A"
        )
        div = info.get("dividendYield")
        div_str = f"{div * 100:.2f}%" if div else "0.0%"
        eps = (
            round(info.get("trailingEps"), 2)
            if info.get("trailingEps")
            else "N/A"
        )
        return cap_str, pe, pb, div_str, eps
    except Exception:
        return "N/A", "N/A", "N/A", "N/A", "N/A"


def calculate_indicators(df):
    if len(df) < 34:
        return ("N/A",) * 17

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs_val = gain / (loss + 1e-9)
    rsi = round(100 - (100 / (1 + rs_val.iloc[-1])), 1)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = round(ema12.iloc[-1] - ema26.iloc[-1], 2)
    ema21 = round(df["close"].ewm(span=21, adjust=False).mean().iloc[-1], 2)

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    bb_upper = round(sma20.iloc[-1] + (std20.iloc[-1] * 2), 2)
    bb_lower = round(sma20.iloc[-1] - (std20.iloc[-1] * 2), 2)

    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp.diff() < 0, 0).rolling(14).sum()
    mfi = round(100 - (100 / (1 + (pos_mf / (neg_mf + 1e-9)).iloc[-1])), 1)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
        axis=1
    )
    atr = round(true_range.rolling(14).mean().iloc[-1], 2)

    vol_mean_20 = df["volume"].rolling(20).mean().iloc[-1]
    rvol = (
        round(df["volume"].iloc[-1] / vol_mean_20, 2) if vol_mean_20 > 0 else 1.0
    )

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    k_line = 100 * ((df["close"] - low14) / (high14 - low14 + 1e-9))
    d_line = k_line.rolling(3).mean()
    stoch_k = round(k_line.iloc[-1], 1) if not pd.isna(k_line.iloc[-1]) else "N/A"
    stoch_d = round(d_line.iloc[-1], 1) if not pd.isna(d_line.iloc[-1]) else "N/A"

    sma50 = (
        round(df["close"].rolling(50).mean().iloc[-1], 2)
        if len(df) >= 50
        else "N/A"
    )
    sma200 = (
        round(df["close"].rolling(200).mean().iloc[-1], 2)
        if len(df) >= 200
        else "N/A"
    )

    sma_tp = tp.rolling(20).mean()
    mean_dev = (tp - sma_tp).abs().rolling(20).mean()
    cci = (
        round(((tp - sma_tp) / (0.015 * mean_dev + 1e-9)).iloc[-1], 1)
        if not pd.isna(((tp - sma_tp) / (0.015 * mean_dev)).iloc[-1])
        else "N/A"
    )
    willr = (
        round(
            (-100 * (high14 - df["close"]) / (high14 - low14 + 1e-9)).iloc[-1],
            1,
        )
        if not pd.isna((-100 * (high14 - df["close"]) / (high14 - low14)).iloc[-1])
        else "N/A"
    )
    roc = (
        round(
            (
                (
                    (df["close"] - df["close"].shift(12))
                    / (df["close"].shift(12) + 1e-9)
                )
                * 100
            ).iloc[-1],
            2,
        )
        if not pd.isna(
            (
                ((df["close"] - df["close"].shift(12)) / df["close"].shift(12))
                * 100
            ).iloc[-1]
        )
        else "N/A"
    )

    mp = (df["high"] + df["low"]) / 2
    ao = (
        round(
            (mp.rolling(5).mean() - mp.rolling(34).mean()).iloc[-1], 2
        )
        if not pd.isna(
            (mp.rolling(5).mean() - mp.rolling(34).mean()).iloc[-1]
        )
        else "N/A"
    )
    vwap = (
        round(
            (
                (df["close"] * df["volume"]).cumsum()
                / (df["volume"].cumsum() + 1e-9)
            ).iloc[-1],
            2,
        )
        if not pd.isna(
            (
                (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            ).iloc[-1]
        )
        else "N/A"
    )

    return (
        rsi,
        macd,
        ema21,
        atr,
        rvol,
        bb_upper,
        bb_lower,
        mfi,
        stoch_k,
        stoch_d,
        sma50,
        sma200,
        cci,
        willr,
        roc,
        ao,
        vwap,
    )


def build_daily_snapshot():
    tz_ny = pytz.timezone("America/New_York")
    now_ny = datetime.datetime.now(tz_ny)
    date_str = now_ny.strftime("%Y-%m-%d")

    snapshot = {
        "date": date_str,
        "timestamp": now_ny.strftime("%Y-%m-%d %H:%M:%S EST"),
        "data": {},
    }

    for symbol in SYMBOLS:
        try:
            t_obj = yf.Ticker(symbol)
            df_intra = t_obj.history(period="2d", interval="5m", prepost=True)
            if df_intra.empty:
                continue

            if df_intra.index.tz is not None:
                df_intra.index = df_intra.index.tz_convert(None)

            curr = float(df_intra["Close"].iloc[-1])
            latest_time = df_intra.index[-1]
            today_df = df_intra[df_intra.index.date == latest_time.date()]

            day_high = (
                round(float(today_df["High"].max()), 2)
                if not today_df.empty
                else round(curr, 2)
            )
            day_low = (
                round(float(today_df["Low"].min()), 2)
                if not today_df.empty
                else round(curr, 2)
            )

            def get_price_ago(minutes):
                target_time = latest_time - pd.Timedelta(minutes=minutes)
                subset = df_intra.loc[df_intra.index <= target_time]
                return subset["Close"].iloc[-1] if not subset.empty else None

            p_10m = get_price_ago(10)
            p_30m = get_price_ago(30)
            p_1h = get_price_ago(60)

            chg_10m = (
                round(((curr - p_10m) / p_10m) * 100, 2)
                if p_10m is not None
                else 0.0
            )
            chg_30m = (
                round(((curr - p_30m) / p_30m) * 100, 2)
                if p_30m is not None
                else 0.0
            )
            chg_1h = (
                round(((curr - p_1h) / p_1h) * 100, 2)
                if p_1h is not None
                else 0.0
            )

            yf_df = t_obj.history(period="1y")
            if yf_df.empty:
                continue
            df = yf_df.rename(
                columns={
                    "Close": "close",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                }
            )

            info = t_obj.info
            prev_close = info.get("previousClose") or info.get(
                "regularMarketPreviousClose"
            )
            if not prev_close:
                prev_close = df["close"].iloc[-2] if len(df) >= 2 else curr

            open_5d = (
                df["open"].iloc[-5] if len(df) >= 5 else df["open"].iloc[0]
            )
            perc_5d = round(((curr - open_5d) / open_5d) * 100, 2)

            (
                rsi,
                macd,
                ema21,
                atr,
                rvol,
                bb_up,
                bb_low,
                mfi,
                stoch_k,
                stoch_d,
                sma50,
                sma200,
                cci,
                willr,
                roc,
                ao,
                vwap,
            ) = calculate_indicators(df)
            cap, pe, pb, div, eps = get_fundamentals(symbol)

            signal = (
                "Buy"
                if (rsi != "N/A" and mfi != "N/A" and rsi < 40 and mfi > 30)
                else (
                    "Sell"
                    if (
                        rsi != "N/A"
                        and mfi != "N/A"
                        and rsi > 65
                        and mfi < 70
                    )
                    else "Hold"
                )
            )
            target_entry = round(bb_low, 2) if signal == "Buy" else round(curr, 2)

            if isinstance(atr, (int, float)) and atr > 0:
                tp_min = round(curr + (1.5 * atr), 2)
                tp_max = round(curr + (3.0 * atr), 2)
                stop_loss = round(curr - (1.5 * atr), 2)
            else:
                tp_min = round(curr * 1.03, 2)
                tp_max = round(curr * 1.07, 2)
                stop_loss = round(curr * 0.95, 2)

            snapshot["data"][symbol] = {
                "Symbol": symbol,
                "Price": round(curr, 2),
                "Day High": day_high,
                "Day Low": day_low,
                "Daily %": round(((curr - prev_close) / prev_close) * 100, 2),
                "10m %": chg_10m,
                "30m %": chg_30m,
                "1h %": chg_1h,
                "5D %": perc_5d,
                "Market Cap": cap,
                "P/E": pe,
                "P/B": pb,
                "Div Yield": div,
                "EPS": eps,
                "RSI": rsi,
                "MACD": macd,
                "CCI": cci,
                "Williams %R": willr,
                "ROC": roc,
                "Awesome Oscillator": ao,
                "VWAP": vwap,
                "ATR": atr,
                "RVOL": rvol,
                "EMA 21": ema21,
                "Stoch %K": stoch_k,
                "Stoch %D": stoch_d,
                "SMA 50": sma50,
                "SMA 200": sma200,
                "MFI": mfi,
                "BB_Up": bb_up,
                "BB_Low": bb_low,
                "Signal": signal,
                "Target Entry": target_entry,
                "TP Min": tp_min,
                "TP Max": tp_max,
                "Stop Loss": stop_loss,
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    return snapshot


def update_database():
    new_data = build_daily_snapshot()
    if not new_data["data"]:
        print("No data collected.")
        return

    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except Exception:
                db = []
    else:
        db = []

    # הסרת רישום קיים לאותו תאריך (למניעת כפילויות בהרצות חוזרות)
    db = [entry for entry in db if entry.get("date") != new_data["date"]]
    db.append(new_data)

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(
        f"[{new_data['timestamp']}] Database updated successfully with {len(new_data['data'])} tickers."
    )


if __name__ == "__main__":
    update_database()