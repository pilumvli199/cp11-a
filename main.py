#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v4.1 (Multiple EMAs + Better Analysis)
import os, re, asyncio, aiohttp, traceback, numpy as np, json
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
def fmt_price(p): return f"{p:.6f}" if abs(p)<1 else f"{p:.2f}"

# Enhanced EMA calculation
def ema(values, period):
    if len(values) < period: return []
    k = 2 / (period + 1)
    prev = sum(values[:period]) / period
    arr = [None] * (period - 1) + [prev]
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        arr.append(prev)
    return arr

# Multiple EMAs calculation
def calculate_multiple_emas(closes):
    ema_periods = [5, 8, 9, 13, 20, 21, 34, 50, 89, 144, 200]
    emas = {}
    
    for period in ema_periods:
        ema_values = ema(closes, period)
        if ema_values:
            emas[f'ema_{period}'] = ema_values[-1] if ema_values[-1] is not None else None
    
    return emas

# EMA trend analysis
def analyze_ema_trend(emas, current_price):
    trend_score = 0
    signals = []
    
    # Short-term EMAs (5, 8, 9, 13)
    short_emas = [emas.get(f'ema_{p}') for p in [5, 8, 9, 13] if emas.get(f'ema_{p}') is not None]
    # Medium-term EMAs (20, 21, 34, 50)
    medium_emas = [emas.get(f'ema_{p}') for p in [20, 21, 34, 50] if emas.get(f'ema_{p}') is not None]
    # Long-term EMAs (89, 144, 200)
    long_emas = [emas.get(f'ema_{p}') for p in [89, 144, 200] if emas.get(f'ema_{p}') is not None]
    
    # Price above/below EMAs
    if short_emas:
        short_avg = sum(short_emas) / len(short_emas)
        if current_price > short_avg:
            trend_score += 2
            signals.append("price above short EMAs")
        else:
            trend_score -= 2
            signals.append("price below short EMAs")
    
    if medium_emas:
        medium_avg = sum(medium_emas) / len(medium_emas)
        if current_price > medium_avg:
            trend_score += 1
            signals.append("price above medium EMAs")
        else:
            trend_score -= 1
            signals.append("price below medium EMAs")
    
    if long_emas:
        long_avg = sum(long_emas) / len(long_emas)
        if current_price > long_avg:
            trend_score += 1
            signals.append("price above long EMAs")
        else:
            trend_score -= 1
            signals.append("price below long EMAs")
    
    # EMA alignment check
    if emas.get('ema_9') and emas.get('ema_21'):
        if emas['ema_9'] > emas['ema_21']:
            trend_score += 1
            signals.append("EMA9 > EMA21")
        else:
            trend_score -= 1
            signals.append("EMA9 < EMA21")
    
    if emas.get('ema_21') and emas.get('ema_50'):
        if emas['ema_21'] > emas['ema_50']:
            trend_score += 1
            signals.append("EMA21 > EMA50")
        else:
            trend_score -= 1
            signals.append("EMA21 < EMA50")
    
    if emas.get('ema_50') and emas.get('ema_200'):
        if emas['ema_50'] > emas['ema_200']:
            trend_score += 2
            signals.append("EMA50 > EMA200 (Golden Cross)")
        else:
            trend_score -= 2
            signals.append("EMA50 < EMA200 (Death Cross)")
    
    return trend_score, signals

# Enhanced horizontal levels
def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    pts = closes[-lookback:] + highs[-lookback:] + lows[-lookback:]
    lvls = []
    for p in pts:
        found = False
        for lv in lvls:
            if abs((lv["price"] - p) / p) < binsize:
                lv["count"] += 1
                lv["price"] = (lv["price"] * (lv["count"] - 1) + p) / lv["count"]
                found = True
                break
        if not found:
            lvls.append({"price": p, "count": 1})
    
    lvls.sort(key=lambda x: -x["count"])
    return [lv["price"] for lv in lvls[:5]]

# Enhanced trade analysis with multiple EMAs
def analyze_trade_logic(candles, rr_min=1.5):
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    
    if len(closes) < 50:
        return {"side": "none", "confidence": 0, "reason": "not enough data"}
    
    current_price = closes[-1]
    emas = calculate_multiple_emas(closes)
    trend_score, trend_signals = analyze_ema_trend(emas, current_price)
    
    # Support and resistance levels
    lvls = horizontal_levels(closes, highs, lows)
    sup = max([lv for lv in lvls if lv < current_price], default=None)
    res = min([lv for lv in lvls if lv > current_price], default=None)
    
    # Base confidence
    base_conf = 50
    
    # Trend-based confidence adjustment
    if trend_score >= 4:
        base_conf += 20
    elif trend_score >= 2:
        base_conf += 10
    elif trend_score <= -4:
        base_conf += 20  # Strong bearish trend
    elif trend_score <= -2:
        base_conf += 10  # Bearish trend
    else:
        base_conf -= 10  # No clear trend
    
    reasons = trend_signals[:3]  # Limit to top 3 signals
    
    # Bullish setup
    if trend_score >= 2 and sup:
        entry = current_price
        stop = sup * 0.997 if sup else current_price * 0.98
        target = res if res else current_price * 1.05
        
        rr = (target - entry) / (entry - stop) if entry > stop else 0
        
        if rr >= rr_min:
            confidence = min(95, base_conf + (trend_score * 2))
            return {
                "side": "BUY",
                "entry": entry,
                "sl": stop,
                "tp": target,
                "confidence": confidence,
                "reason": "; ".join(reasons),
                "trend_score": trend_score,
                "rr": round(rr, 2)
            }
    
    # Bearish setup
    elif trend_score <= -2 and res:
        entry = current_price
        stop = res * 1.003 if res else current_price * 1.02
        target = sup if sup else current_price * 0.95
        
        rr = (entry - target) / (stop - entry) if stop > entry else 0
        
        if rr >= rr_min:
            confidence = min(95, base_conf + (abs(trend_score) * 2))
            return {
                "side": "SELL",
                "entry": entry,
                "sl": stop,
                "tp": target,
                "confidence": confidence,
                "reason": "; ".join(reasons),
                "trend_score": trend_score,
                "rr": round(rr, 2)
            }
    
    return {
        "side": "none", 
        "confidence": base_conf, 
        "reason": "; ".join(reasons) if reasons else "no clear setup",
        "trend_score": trend_score
    }

def multi_tf_confirmation(c30, c1h):
    s30 = analyze_trade_logic(c30)
    s1h = analyze_trade_logic(c1h)
    
    if s30["side"] == "none" or s1h["side"] == "none":
        return {"side": "none", "confidence": 0, "reason": "no alignment"}
    
    if s30["side"] == s1h["side"]:
        # Both timeframes agree
        combined_conf = min(100, (s30["confidence"] + s1h["confidence"]) // 2 + 15)
        s30["confidence"] = combined_conf
        s30["reason"] += f"; 1H aligned (trend: {s1h.get('trend_score', 0)})"
        return s30
    
    return {"side": "none", "confidence": 0, "reason": "timeframe conflict"}

# Enhanced chart with multiple EMAs
def plot_signal_chart(symbol, candles, signal):
    dates = [datetime.utcfromtimestamp(int(x[0]) / 1000) for x in candles]
    closes = [float(x[4]) for x in candles]
    highs = [float(x[2]) for x in candles]
    lows = [float(x[3]) for x in candles]
    opens = [float(x[1]) for x in candles]
    
    x = date2num(dates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Candlestick chart
    for i, (xi, o, h, l, c) in enumerate(zip(x, opens, highs, lows, closes)):
        color = "green" if c >= o else "red"
        ax1.plot([xi, xi], [l, h], color=color, linewidth=0.8)
        ax1.plot([xi, xi], [o, c], color=color, linewidth=3)
    
    # Calculate and plot multiple EMAs
    emas = calculate_multiple_emas(closes)
    ema_colors = {
        'ema_9': 'blue', 'ema_21': 'orange', 'ema_50': 'purple', 
        'ema_89': 'brown', 'ema_200': 'red'
    }
    
    for period in [9, 21, 50, 89, 200]:
        ema_key = f'ema_{period}'
        if ema_key in emas:
            ema_values = ema(closes, period)
            if len(ema_values) == len(x):
                valid_indices = [i for i, val in enumerate(ema_values) if val is not None]
                if valid_indices:
                    ax1.plot([x[i] for i in valid_indices], 
                           [ema_values[i] for i in valid_indices], 
                           color=ema_colors.get(ema_key, 'gray'), 
                           label=f'EMA{period}', linewidth=1.5, alpha=0.8)
    
    # Signal lines
    if signal.get("entry"):
        ax1.axhline(signal["entry"], color="blue", linestyle="--", 
                   label=f"Entry {fmt_price(signal['entry'])}", alpha=0.8)
    if signal.get("sl"):
        ax1.axhline(signal["sl"], color="red", linestyle="--", 
                   label=f"SL {fmt_price(signal['sl'])}", alpha=0.8)
    if signal.get("tp"):
        ax1.axhline(signal["tp"], color="green", linestyle="--", 
                   label=f"TP {fmt_price(signal['tp'])}", alpha=0.8)
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(f"{symbol} - {signal.get('side', 'NO')} Signal | "
                 f"Conf: {signal.get('confidence', 0)}% | "
                 f"Trend Score: {signal.get('trend_score', 0)} | "
                 f"R/R: {signal.get('rr', 0)}", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Volume chart (placeholder - using closes as proxy)
    volumes = [abs(closes[i] - opens[i]) * 1000 for i in range(len(closes))]  # Proxy volume
    ax2.bar(x, volumes, color=['green' if closes[i] >= opens[i] else 'red' for i in range(len(closes))], 
           alpha=0.6, width=0.8)
    ax2.set_title("Volume (Proxy)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception as e:
        print(f"Fetch error: {e}")
        return None

# ---------------- Enhanced AI Analysis ----------------
async def ask_openai_for_signals(symbol, emas, signal_data):
    if not client:
        return None
    
    # Prepare comprehensive market data for AI
    market_summary = {
        "symbol": symbol,
        "current_price": signal_data.get("entry", 0),
        "trend_score": signal_data.get("trend_score", 0),
        "ema_analysis": {
            "ema_9": emas.get("ema_9"),
            "ema_21": emas.get("ema_21"),
            "ema_50": emas.get("ema_50"),
            "ema_89": emas.get("ema_89"),
            "ema_200": emas.get("ema_200")
        },
        "local_signal": {
            "side": signal_data.get("side"),
            "confidence": signal_data.get("confidence"),
            "rr_ratio": signal_data.get("rr"),
            "reason": signal_data.get("reason")
        }
    }
    
    system_prompt = """You are an expert crypto trader with deep knowledge of technical analysis and market psychology. 
    Analyze the provided market data and give your professional opinion on the trade setup.
    
    Focus on:
    1. EMA alignment and trend strength
    2. Risk-reward ratio assessment
    3. Market structure analysis
    4. Entry timing optimization
    
    Respond with: CONFIRM/REJECT followed by confidence % and brief reasoning."""
    
    user_prompt = f"""Market Analysis for {symbol}:
    
    {json.dumps(market_summary, indent=2)}
    
    Current local analysis suggests: {signal_data.get('side', 'NONE')} with {signal_data.get('confidence', 0)}% confidence.
    
    Do you CONFIRM or REJECT this signal? Provide confidence % (0-100) and key reasoning."""
    
    try:
        loop = asyncio.get_running_loop()
        
        def call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
        
        resp = await loop.run_in_executor(None, call)
        ai_response = resp.choices[0].message.content.strip()
        print(f"AI Response for {symbol}: {ai_response[:100]}...")
        return ai_response
        
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
            os.unlink(path)  # Clean up temp file
        except:
            pass

# ---------------- Main Enhanced Loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup = f"🤖 Enhanced Bot Started! • {len(SYMBOLS)} symbols • Multiple EMAs: 5,8,9,13,20,21,34,50,89,144,200 • Poll: {POLL_INTERVAL}s"
        print(startup)
        await send_text(session, startup)
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration} @ {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*50}")
            
            signals_found = 0
            
            for sym in SYMBOLS:
                try:
                    print(f"\n📊 Analyzing {sym}...")
                    
                    # Fetch candle data
                    c30 = await fetch_json(session, CANDLE_URL.format(symbol=sym))
                    c1h = await fetch_json(session, CANDLE_URL_1H.format(symbol=sym))
                    
                    if not c30 or not c1h:
                        print(f"❌ Failed to fetch data for {sym}")
                        continue
                    
                    # Process candles
                    candles_30m = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c30]
                    candles_1h = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c1h]
                    
                    # Multi-timeframe analysis
                    local_signal = multi_tf_confirmation(candles_30m, candles_1h)
                    
                    if local_signal["side"] != "none" and local_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                        # Calculate EMAs for AI analysis
                        closes = [float(x[4]) for x in c30]
                        emas = calculate_multiple_emas(closes)
                        
                        # Get AI confirmation
                        ai_response = await ask_openai_for_signals(sym, emas, local_signal)
                        ai_boost = 0
                        
                        if ai_response and "CONFIRM" in ai_response.upper():
                            ai_boost = 10
                            print(f"✅ AI confirmed signal for {sym}")
                        elif ai_response and "REJECT" in ai_response.upper():
                            ai_boost = -15
                            print(f"❌ AI rejected signal for {sym}")
                        
                        # Final confidence calculation
                        final_confidence = min(95, local_signal["confidence"] + ai_boost)
                        
                        if final_confidence >= SIGNAL_CONF_THRESHOLD:
                            signals_found += 1
                            
                            # Prepare enhanced message
                            msg = f"🚀 {sym} {local_signal['side']} SIGNAL\n"
                            msg += f"📈 Entry: {fmt_price(local_signal['entry'])}\n"
                            msg += f"🛑 Stop Loss: {fmt_price(local_signal['sl'])}\n"
                            msg += f"🎯 Take Profit: {fmt_price(local_signal['tp'])}\n"
                            msg += f"📊 Confidence: {final_confidence}%\n"
                            msg += f"⚖️ Risk/Reward: 1:{local_signal.get('rr', 0)}\n"
                            msg += f"📉 Trend Score: {local_signal.get('trend_score', 0)}\n"
                            msg += f"🧠 AI: {'✅' if ai_boost > 0 else '❌' if ai_boost < 0 else '➖'}\n"
                            msg += f"💡 Reason: {local_signal['reason'][:100]}..."
                            
                            # Generate enhanced chart
                            chart_path = plot_signal_chart(sym, c30, local_signal)
                            await send_photo(session, msg, chart_path)
                            
                            print(f"⚡ SIGNAL SENT: {sym} {local_signal['side']} | Conf: {final_confidence}%")
                        else:
                            print(f"📉 {sym}: Signal confidence too low ({final_confidence}%)")
                    else:
                        trend_info = local_signal.get('trend_score', 0)
                        print(f"📊 {sym}: No signal (Conf: {local_signal['confidence']}%, Trend: {trend_info})")
                
                except Exception as e:
                    print(f"❌ Error analyzing {sym}: {e}")
                    traceback.print_exc()
            
            print(f"\n📊 Iteration {iteration} complete: {signals_found} signals found")
            
            # Send summary if no signals found
            if signals_found == 0:
                summary_msg = f"📊 Scan #{iteration} Complete\n✅ {len(SYMBOLS)} symbols analyzed\n📈 0 signals above {int(SIGNAL_CONF_THRESHOLD)}% threshold\n⏰ Next scan in {POLL_INTERVAL//60}min"
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
