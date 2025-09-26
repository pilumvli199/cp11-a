# ===== patched main.py =====
#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v4.1-patch (EMA9 + EMA20, robust candle parsing)
# Patched by assistant: replaced non-ASCII characters in function names and improved safety.

import os
import re
import asyncio
import aiohttp
import traceback
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=200"
CANDLE_URL_1H = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=200"

# ---------------- Utils ----------------

def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"

# Robust EMA (returns list aligned with input length; initial (period-1) entries are None)
def ema(values, period):
    if not values or len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [None] * (period - 1)
    # SMA for first EMA value
    sma = sum(values[:period]) / period
    ema_vals.append(sma)
    prev = sma
    for i in range(period, len(values)):
        v = values[i]
        prev = v * k + prev * (1 - k)
        ema_vals.append(prev)
    # If input length equals period, ema_vals length equals len(values)
    return ema_vals

# Only EMA 9 and EMA 20 as requested
def calculate_emas_9_20(closes):
    """
    returns dict with keys 'ema_9' and 'ema_20' values aligned to last candle (or None)
    """
    out = {}
    try:
        e9 = ema(closes, 9)
        e20 = ema(closes, 20)
        out['ema_9'] = e9[-1] if e9 else None
        out['ema_20'] = e20[-1] if e20 else None
    except Exception:
        out['ema_9'] = None
        out['ema_20'] = None
    return out

# normalize raw kline rows into dicts: [o,h,l,close,volume,ts]
def normalize_klines(raw_klines):
    """
    Accepts list of kline rows (binance raw) where each row is list-like.
    Outputs list of dicts: {"open":float,"high":float,"low":float,"close":float,"volume":float,"ts":int}
    Skips malformed rows gracefully.
    """
    out = []
    for row in raw_klines or []:
        try:
            # Binance kline: [openTime, open, high, low, close, volume, ...]
            if len(row) >= 6:
                ts = int(row[0]) if isinstance(row[0], (int, float, str)) else None
                o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[5])
                out.append({"open": o, "high": h, "low": l, "close": c, "volume": v, "ts": ts})
            else:
                # maybe a shorter processed candle like [o,h,l,close]
                if len(row) >= 4:
                    o = float(row[0]); h = float(row[1]); l = float(row[2]); c = float(row[3])
                    out.append({"open": o, "high": h, "low": l, "close": c, "volume": 0.0, "ts": None})
                else:
                    # skip
                    continue
        except Exception:
            # skip malformed row
            continue
    return out

# Enhanced horizontal levels (safe handling)
def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    try:
        length = min(len(closes), lookback)
        pts = closes[-length:] + highs[-length:] + lows[-length:]
    except Exception:
        return []
    levels = []
    for p in pts:
        if p is None or p == 0:
            continue
        found = False
        for lv in levels:
            try:
                if abs((lv["price"] - p) / p) < binsize:
                    lv["count"] += 1
                    lv["price"] = (lv["price"] * (lv["count"] - 1) + p) / lv["count"]
                    found = True
                    break
            except Exception:
                continue
        if not found:
            levels.append({"price": p, "count": 1})
    levels.sort(key=lambda x: -x["count"])
    return [lv["price"] for lv in levels][:5]

# ---------------- Trade logic (robust, uses normalized candles) ----------------
def analyze_trade_logic(raw_candles, rr_min=1.5):
    """
    raw_candles: list of raw kline rows OR normalized candle dicts
    returns trade dict or 'none'
    """
    try:
        # Normalize if needed
        if not raw_candles:
            return {"side": "none", "confidence": 0, "reason": "no candles"}
        # If already normalized (dicts with 'close'), keep; else normalize
        if isinstance(raw_candles[0], dict) and 'close' in raw_candles[0]:
            candles = raw_candles
        else:
            candles = normalize_klines(raw_candles)

        if len(candles) < 60:
            return {"side": "none", "confidence": 0, "reason": "insufficient data"}

        closes = [c["close"] for c in candles if c and "close" in c]
        highs = [c["high"] for c in candles if c and "high" in c]
        lows = [c["low"] for c in candles if c and "low" in c]
        if len(closes) < 60:
            return {"side": "none", "confidence": 0, "reason": "insufficient valid closes"}

        current_price = closes[-1]

        # EMAs (only 9 & 20)
        emas = calculate_emas_9_20(closes)
        ema9 = emas.get('ema_9')
        ema20 = emas.get('ema_20')

        # Basic safety checks
        if ema9 is None or ema20 is None:
            return {"side": "none", "confidence": 0, "reason": "ema values missing"}

        # Trend scoring using EMA 9 & 20
        trend_score = 0
        reasons = []
        if current_price > ema9:
            trend_score += 2; reasons.append("price > EMA9")
        else:
            trend_score -= 1; reasons.append("price <= EMA9")
        if ema9 > ema20:
            trend_score += 2; reasons.append("EMA9 > EMA20 (bullish)")
        else:
            trend_score -= 2; reasons.append("EMA9 <= EMA20 (bearish)")

        # Support/resistance
        lvls = horizontal_levels(closes, highs, lows, lookback=50)
        support = max([lv for lv in lvls if lv < current_price], default=None) if lvls else None
        resistance = min([lv for lv in lvls if lv > current_price], default=None) if lvls else None

        base_conf = 50
        if trend_score >= 3:
            base_conf += 20
        elif trend_score == 2:
            base_conf += 10
        elif trend_score <= -3:
            base_conf += 15
        elif trend_score <= -1:
            base_conf += 5
        else:
            base_conf -= 5

        # Bullish candidate
        if trend_score >= 2 and support:
            entry = float(current_price)
            sl = float(support) * 0.997 if support else entry * 0.985
            tp = float(resistance) if resistance else entry * 1.05
            denom = (entry - sl)
            if denom > 0:
                rr = (tp - entry) / denom
                if rr >= rr_min:
                    confidence = min(95, base_conf + trend_score*2)
                    return {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": confidence,
                        "reason": "; ".join(reasons[:3]),
                        "trend_score": trend_score,
                        "rr": round(rr, 2),
                        "indicators": {"ema_9": ema9, "ema_20": ema20}
                    }

        # Bearish candidate
        if trend_score <= -2 and resistance:
            entry = float(current_price)
            sl = float(resistance) * 1.003 if resistance else entry * 1.015
            tp = float(support) if support else entry * 0.95
            denom = (sl - entry)
            if denom > 0:
                rr = (entry - tp) / denom
                if rr >= rr_min:
                    confidence = min(95, base_conf + abs(trend_score)*2)
                    return {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": confidence,
                        "reason": "; ".join(reasons[:3]),
                        "trend_score": trend_score,
                        "rr": round(rr, 2),
                        "indicators": {"ema_9": ema9, "ema_20": ema20}
                    }

        return {
            "side": "none",
            "confidence": base_conf,
            "reason": "; ".join(reasons) if reasons else "no clear setup",
            "trend_score": trend_score,
            "indicators": {"ema_9": ema9, "ema_20": ema20}
        }

    except Exception as e:
        # Safety net
        print(f"analyze_trade_logic fatal: {e}")
        traceback.print_exc()
        return {"side": "none", "confidence": 0, "reason": f"internal error: {e}"}


def multi_tf_confirmation(c30, c1h):
    """Accepts raw klines for 30m and 1h"""
    s30 = analyze_trade_logic(c30)
    s1h = analyze_trade_logic(c1h)
    if s30.get("side") == "none" or s1h.get("side") == "none":
        return {"side": "none", "confidence": 0, "reason": "no alignment"}
    if s30["side"] == s1h["side"]:
        combined_conf = min(100, int((s30.get("confidence",0) + s1h.get("confidence",0))/2) + 15)
        s30["confidence"] = combined_conf
        s30["reason"] = s30.get("reason", "") + f"; 1H aligned (trend: {s1h.get('trend_score',0)})"
        return s30
    return {"side": "none", "confidence": 0, "reason": "timeframe conflict"}

# ---------------- Charting ----------------
def plot_signal_chart(symbol, raw_candles, signal):
    candles = normalize_klines(raw_candles)
    if not candles:
        # fallback simple text-image
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6,2)); plt.text(0.5,0.5,"No data",ha='center'); plt.axis('off'); plt.savefig(tmp.name); plt.close()
        return tmp.name
    dates = [datetime.utcfromtimestamp(c["ts"]/1000) if c["ts"] else datetime.utcnow() for c in candles]
    closes = [c["close"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    x = date2num(dates)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,8), gridspec_kw={'height_ratios':[3,1]})
    # candlestick-ish
    for i, (xi, o, h, l, c) in enumerate(zip(x, opens, highs, lows, closes)):
        color = "green" if c >= o else "red"
        ax1.vlines(xi, l, h, color=color, linewidth=0.8)
        ax1.plot([xi, xi], [o, c], color=color, linewidth=3)

    # plot EMA9 & EMA20 over closes if available as full series
    try:
        e9_series = ema(closes, 9)
        e20_series = ema(closes, 20)
        # only plot if lengths match x
        if e9_series and len(e9_series) == len(x):
            ax1.plot(x, e9_series, label='EMA9', linewidth=1.2)
        if e20_series and len(e20_series) == len(x):
            ax1.plot(x, e20_series, label='EMA20', linewidth=1.2)
    except Exception:
        pass

    if signal.get('entry'): ax1.axhline(signal['entry'], linestyle='--', label=f"Entry {fmt_price(signal['entry'])}")
    if signal.get('sl'): ax1.axhline(signal['sl'], linestyle='--', label=f"SL {fmt_price(signal['sl'])}")
    if signal.get('tp'): ax1.axhline(signal['tp'], linestyle='--', label=f"TP {fmt_price(signal['tp'])}")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # volume proxy
    volumes = [abs(closes[i]-opens[i])*1000 for i in range(len(closes))]
    ax2.bar(x, volumes, alpha=0.6)
    ax2.set_title("Volume (proxy)")
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                # return None but print quick info for debugging
                text = await r.text()
                print(f"fetch_json {url} -> {r.status} : {text[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return None

# ---------------- Enhanced AI Analysis ----------------
async def ask_openai_for_signals(symbol, emas, signal_data):
    if not client:
        return None
    market_summary = {
        "symbol": symbol,
        "current_price": signal_data.get("entry", 0),
        "trend_score": signal_data.get("trend_score", 0),
        "ema_9": emas.get("ema_9"),
        "ema_20": emas.get("ema_20"),
        "local_signal": {
            "side": signal_data.get("side"),
            "confidence": signal_data.get("confidence"),
            "rr": signal_data.get("rr"),
            "reason": signal_data.get("reason")
        }
    }
    system_prompt = ("You are an expert crypto trader. Analyze the provided market summary and either CONFIRM or REJECT "
                     "the proposed trade. Provide a short CONF:% and brief reason.")
    user_prompt = f"Market summary:\n{json.dumps(market_summary, indent=2)}\nRespond: CONFIRM/REJECT - CONF:xx% - REASON: short"
    try:
        loop = asyncio.get_running_loop()
        def call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                max_tokens=200,
                temperature=0.2
            )
        resp = await loop.run_in_executor(None, call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print(f"Telegram text error: {e}")

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption)
        try:
            os.unlink(path)
        except Exception:
            pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", TELEGRAM_CHAT_ID)
            data.add_field("caption", caption)
            data.add_field("photo", f)
            await session.post(url, data=data)
    except Exception as e:
        print(f"Telegram photo error: {e}")
    finally:
        try:
            os.unlink(path)
        except:
            pass

# ---------------- Main Enhanced Loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup = f"🤖 Enhanced Bot Started! • {len(SYMBOLS)} symbols • EMAs: 9 & 20 • Poll: {POLL_INTERVAL}s"
        print(startup)
        await send_text(session, startup)

        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*60}")

            signals_found = 0

            for sym in SYMBOLS:
                try:
                    print(f"\n📊 Analyzing {sym}...")

                    # Fetch candle data (raw binance klines)
                    c30 = await fetch_json(session, CANDLE_URL.format(symbol=sym))
                    c1h = await fetch_json(session, CANDLE_URL_1H.format(symbol=sym))

                    if not c30 or not c1h:
                        print(f"❌ Failed to fetch data for {sym} (30m:{bool(c30)} 1h:{bool(c1h)})")
                        continue

                    # Use raw klines directly (normalize inside analysis)
                    local_signal = multi_tf_confirmation(c30, c1h)

                    if local_signal.get("side") != "none" and local_signal.get("confidence", 0) >= SIGNAL_CONF_THRESHOLD:
                        # Compute EMAs from closes for AI & chart
                        # normalize and extract closes safely
                        normalized_30 = normalize_klines(c30)
                        closes = [c["close"] for c in normalized_30]
                        emas = calculate_emas_9_20(closes)

                        # Ask AI for confirmation
                        ai_response = await ask_openai_for_signals(sym, emas, local_signal)
                        ai_boost = 0
                        if ai_response and "CONFIRM" in ai_response.upper():
                            ai_boost = 10
                            print(f"✅ AI confirmed signal for {sym}")
                        elif ai_response and "REJECT" in ai_response.upper():
                            ai_boost = -15
                            print(f"❌ AI rejected signal for {sym}")

                        final_confidence = min(95, int(local_signal.get("confidence",0) + ai_boost))

                        if final_confidence >= SIGNAL_CONF_THRESHOLD:
                            signals_found += 1
                            msg = (f"🚀 {sym} {local_signal['side']} SIGNAL\n"
                                   f"📈 Entry: {fmt_price(local_signal['entry'])}\n"
                                   f"🛑 Stop Loss: {fmt_price(local_signal['sl'])}\n"
                                   f"🎯 Take Profit: {fmt_price(local_signal['tp'])}\n"
                                   f"📊 Confidence: {final_confidence}%\n"
                                   f"⚖️ Risk/Reward: 1:{local_signal.get('rr',0)}\n"
                                   f"💡 Reason: {local_signal.get('reason','')[:140]}")

                            chart_path = plot_signal_chart(sym, c30, local_signal)
                            await send_photo(session, msg, chart_path)
                            print(f"⚡ SIGNAL SENT: {sym} {local_signal['side']} | Conf: {final_confidence}%")
                        else:
                            print(f"📉 {sym}: Signal confidence too low ({final_confidence}%)")
                    else:
                        print(f"📊 {sym}: No signal (Conf: {local_signal.get('confidence',0)}%, Reason: {local_signal.get('reason')})")

                except Exception as e:
                    print(f"❌ Error analyzing {sym}: {e}")
                    traceback.print_exc()

            print(f"\n📊 Iteration {iteration} complete: {signals_found} signals found")
            if signals_found == 0:
                summary_msg = (f"📊 Scan #{iteration} Complete\n"
                               f"✅ {len(SYMBOLS)} symbols analyzed\n"
                               f"📈 0 signals above {int(SIGNAL_CONF_THRESHOLD)}% threshold\n"
                               f"⏰ Next scan in {POLL_INTERVAL//60}min")
                await send_text(session, summary_msg)

            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
