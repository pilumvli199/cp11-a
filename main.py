# main_updated_trades.py
# Phase-5 upgraded: 30m 100 candles, early-trend alerts, Entry/SL/Targets (1R/2R/3R), chart+text to Telegram

import os
import asyncio
import aiohttp
import traceback
from datetime import datetime
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 60.0))

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; send_text skipped.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}) as r:
            if r.status != 200:
                body = await r.text()
                print("Telegram send_text failed:", r.status, body[:300])
    except Exception as e:
        print("send_text exception:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; send_photo skipped.")
        try:
            os.remove(path)
        except:
            pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("photo", f, filename="chart.png", content_type="image/png")
            async with session.post(url, data=data, timeout=60) as r:
                if r.status != 200:
                    body = await r.text()
                    print("Telegram send_photo failed:", r.status, body[:300])
    except Exception as e:
        print("send_photo exception:", e)
    finally:
        try:
            os.remove(path)
        except:
            pass

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                try:
                    txt = await r.text()
                except:
                    txt = "<no-body>"
                print(f"fetch_json {url} status={r.status} body={txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json exception for", url, e)
        return None

async def fetch_data(session, symbol):
    t = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    c = await fetch_json(session, CANDLE_URL.format(symbol=symbol))
    out = {}
    if t:
        try:
            out["price"] = float(t.get("lastPrice", 0))
            out["volume"] = float(t.get("volume", 0))
        except:
            out["price"] = None
            out["volume"] = None
    if isinstance(c, list):
        try:
            # store [o,h,l,c,vol]
            out["candles"] = [
                [float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in c
            ]
            out["times"] = [int(x[0]) // 1000 for x in c]
        except Exception as e:
            print("candle parse error", symbol, e)
            out["candles"] = None
            out["times"] = None
    return out

# ---------------- Price-action, early-trend, Entry/SL/Targets ----------------
def compute_levels(candles, lookback=50):
    if not candles or len(candles) < 3:
        return (None, None, None)
    arr = candles[-lookback:] if len(candles) >= lookback else candles
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k]) / k if highs else None
    sup = sum(lows[:k]) / k if lows else None
    mid = (res + sup) / 2 if res is not None and sup is not None else None
    return sup, res, mid

def is_bullish_engulfing(prev, last):
    return (last[3] > last[0]) and (prev[3] < prev[0]) and (last[3] >= prev[0]) and (last[0] <= prev[3])

def is_bearish_engulfing(prev, last):
    return (last[3] < last[0]) and (prev[3] > prev[0]) and (last[0] >= prev[3]) and (last[3] <= prev[0])

def slope_positive(closes, n=12, thresh=0.0):
    if len(closes) < n:
        return None
    seg = np.array(closes[-n:])
    xs = np.arange(len(seg))
    m, b = np.polyfit(xs, seg, 1)
    return m  # raw slope (use sign)

def last_breakout_recent(candles, level, side="above", lookback=3):
    """
    check if within last `lookback` candles close crossed above (or below) level
    """
    if not candles or level is None:
        return False
    recent = candles[-lookback:]
    if side == "above":
        return any(c[3] > level for c in recent)
    else:
        return any(c[3] < level for c in recent)

def nearest_swing_low(candles, lookback=6):
    seg = candles[-lookback:]
    lows = [c[2] for c in seg]
    return min(lows) if lows else None

def nearest_swing_high(candles, lookback=6):
    seg = candles[-lookback:]
    highs = [c[1] for c in seg]
    return max(highs) if highs else None

def compute_entry_sl_targets(symbol, data):
    """
    Returns dict with bias, entry, sl, targets(list), reason, conf
    """
    candles = data.get("candles")
    if not candles or len(candles) < 6:
        return None
    closes = [c[3] for c in candles]
    last = candles[-1]
    prev = candles[-2]
    sup, res, mid = compute_levels(candles, lookback=50)
    reason_parts = []
    score = 50

    # basic PA
    if res and last[3] > res:
        reason_parts.append("Breakout above resistance")
        score += 18
    if sup and last[3] < sup:
        reason_parts.append("Breakdown below support")
        score += 18

    if is_bullish_engulfing(prev, last):
        reason_parts.append("Bullish Engulfing")
        score += 12
    if is_bearish_engulfing(prev, last):
        reason_parts.append("Bearish Engulfing")
        score += 12

    # slope (early trend)
    m = slope_positive(closes, n=12)
    if m is not None:
        if m > 0:
            reason_parts.append("Positive slope (early uptrend)")
            score += 8
        elif m < 0:
            reason_parts.append("Negative slope (early downtrend)")
            score -= 8

    # recent breakout confirm within last 3 candles
    breakout_recent = False
    breakdown_recent = False
    if res:
        if last_breakout_recent(candles, res, "above"):
            breakout_recent = True
            score += 6
    if sup:
        if last_breakout_recent(candles, sup, "below"):
            breakdown_recent = True
            score += 6

    # volume surge
    vols = [c[4] for c in candles if len(c) >= 5]
    if len(vols) >= 6:
        avgv = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
        if last[4] > 1.8 * avgv:
            reason_parts.append("Volume surge")
            score += 6

    # Determine bias heuristics
    bullish_flags = 0
    bearish_flags = 0
    if res and last[3] > res: bullish_flags += 1
    if is_bullish_engulfing(prev, last): bullish_flags += 1
    if m is not None and m > 0: bullish_flags += 1
    if sup and last[3] < sup: bearish_flags += 1
    if is_bearish_engulfing(prev, last): bearish_flags += 1
    if m is not None and m < 0: bearish_flags += 1

    bias = "NEUTRAL"
    if bullish_flags >= 2 and score >= 60:
        bias = "BUY"
    elif bearish_flags >= 2 and score >= 60:
        bias = "SELL"

    # Entry = last close (you can modify to better entries later)
    entry = last[3]

    # SL: for BUY -> nearest swing low (last 6), or sup (whichever is lower)
    sl = None
    if bias == "BUY":
        swing_low = nearest_swing_low(candles, lookback=6)
        if swing_low:
            # place SL slightly below swing low (0.3% buffer) to avoid noise
            sl = swing_low * 0.997
        elif sup:
            sl = sup * 0.997
        else:
            sl = entry * 0.98  # fallback 2% below
    elif bias == "SELL":
        swing_high = nearest_swing_high(candles, lookback=6)
        if swing_high:
            sl = swing_high * 1.003
        elif res:
            sl = res * 1.003
        else:
            sl = entry * 1.02

    # Targets: 1R,2R,3R (risk-based)
    targets = []
    if sl:
        if bias == "BUY":
            risk = entry - sl
            targets = [entry + risk * r for r in (1, 2, 3)]
        elif bias == "SELL":
            risk = sl - entry
            targets = [entry - risk * r for r in (1, 2, 3)]

    conf = max(0, min(100, int(score)))
    reason = "; ".join(reason_parts) if reason_parts else "No clear PA signal"

    return {
        "symbol": symbol,
        "bias": bias,
        "entry": round(entry, 8) if entry is not None else None,
        "sl": round(sl, 8) if sl is not None else None,
        "targets": [round(t, 8) for t in targets] if targets else [],
        "reason": reason,
        "conf": conf,
        "levels": {"sup": sup, "res": res, "mid": mid},
        "early_trend_slope": m,
        "breakout_recent": breakout_recent,
        "breakdown_recent": breakdown_recent
    }

# ---------------- Chart plotting with markers ----------------
def plot_chart(times, candles, sym, trade_info):
    if not times or not candles or len(times) != len(candles):
        raise ValueError("Insufficient data to plot")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; c_ = [c[3] for c in candles]
    x = date2num(dates)
    fig, ax = plt.subplots(figsize=(9,5), dpi=100)

    width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4
    for xi, oi, hi, li, ci in zip(x, o, h, l, c_):
        col = "white" if ci >= oi else "black"
        ax.vlines(xi, li, hi, color="black", linewidth=0.7)
        rect = plt.Rectangle((xi - width/2, min(oi,ci)), width, max(0.0000001, abs(ci-oi)), facecolor=col, edgecolor="black")
        ax.add_patch(rect)

    levs = trade_info.get("levels", {})
    sup = levs.get("sup"); res = levs.get("res"); mid = levs.get("mid")
    if res is not None:
        ax.axhline(res, linestyle="--", label=f"Res {res:.6f}" if abs(res) < 1 else f"Res {res:.2f}", color="orange")
    if sup is not None:
        ax.axhline(sup, linestyle="--", label=f"Sup {sup:.6f}" if abs(sup) < 1 else f"Sup {sup:.2f}", color="purple")
    if mid is not None:
        ax.axhline(mid, linestyle=":", label=f"Mid {mid:.6f}" if abs(mid) < 1 else f"Mid {mid:.2f}", color="gray")

    # plot trendline (last 20 closes)
    if len(c_) >= 5:
        n = min(20, len(c_))
        xs = np.arange(n)
        ys = np.array(c_[-n:])
        try:
            m, b = np.polyfit(xs, ys, 1)
            yy = m * xs + b
            ax.plot(x[-n:], yy, linestyle='-.', label="Trend (last20)", color="blue")
        except:
            pass

    # Mark entry, SL, targets
    entry = trade_info.get("entry")
    sl = trade_info.get("sl")
    targets = trade_info.get("targets", [])
    last_x = x[-1] if len(x) else None

    if entry is not None:
        # draw a blue dot at last candle x for entry
        ax.scatter([last_x], [entry], color="blue", zorder=5, label="Entry", s=40)
        ax.annotate(f"Entry {entry}", (last_x, entry), textcoords="offset points", xytext=(6,6), fontsize=8, color="blue")

    if sl is not None:
        ax.axhline(sl, linestyle="--", color="red", label=f"SL {sl}")
        ax.annotate(f"SL {sl}", (x[0], sl), textcoords="offset points", xytext=(6,6), fontsize=8, color="red")

    colors = ["green","green","green"]
    for i, t in enumerate(targets):
        ax.axhline(t, linestyle=":", color=colors[i%len(colors)], label=f"T{i+1} {t}")
        ax.annotate(f"T{i+1} {t}", (x[-1], t), textcoords="offset points", xytext=(6,-10*(i+1)), fontsize=8, color=colors[i%len(colors)])

    ax.set_title(f"{sym} 30m — last {len(candles)} candles")
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ---------------- Main loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, f"PA+Trade-Bot online — 30m timeframe, 100 candles, conf≥{SIGNAL_CONF_THRESHOLD}%")
        while True:
            try:
                tasks = [fetch_data(session, s) for s in SYMBOLS]
                results = await asyncio.gather(*tasks)
                market = {s: r for s, r in zip(SYMBOLS, results)}

                # process each symbol locally
                for sym, data in market.items():
                    try:
                        trade = compute_entry_sl_targets(sym, data)
                        if not trade:
                            continue
                        bias = trade["bias"]
                        conf = trade["conf"]
                        if bias in ("BUY", "SELL") and conf >= SIGNAL_CONF_THRESHOLD:
                            # Build message
                            entry = trade["entry"]
                            sl = trade["sl"]
                            targets = trade["targets"]
                            reason = trade["reason"]
                            lvl = trade["levels"]
                            msg_lines = [
                                f"🚨 *{sym}*  — *{bias}*",
                                f"Entry: `{entry}`",
                                f"SL: `{sl}`",
                                f"Targets: " + ", ".join([f"`{t}`" for t in targets]) if targets else "Targets: -",
                                f"Reason: {reason}",
                                f"Confidence: {conf}%",
                                f"Timeframe: 30m (100 candles)",
                                f"Levels: " + ("R: {:.6f} S: {:.6f}".format(lvl['res'], lvl['sup']) if lvl.get('res') and lvl.get('sup') else "N/A")
                            ]
                            text = "\n".join(msg_lines)

                            # plot chart with markers
                            try:
                                chart = plot_chart(data.get("times"), data.get("candles"), sym, trade)
                            except Exception as e:
                                print("plot_chart failed for", sym, e)
                                chart = None

                            # send text then photo
                            await send_text(session, text)
                            if chart:
                                await send_photo(session, text, chart)
                    except Exception as e:
                        print("symbol loop error", sym, e)
                        traceback.print_exc()

                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("main loop exception", e)
                traceback.print_exc()
                await asyncio.sleep(min(60, POLL_INTERVAL))

if __name__ == "__main__":
    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        print("Stopped by user.")
