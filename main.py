#!/usr/bin/env python3
# main.py - Qwen 2.5 VL 72B Crypto Trading Bot (OpenRouter)
# Enhanced with EMA, Price Action, Multi-TF Analysis

import os
import asyncio
import aiohttp
import traceback
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.patches as patches
from tempfile import NamedTemporaryFile
import io
from PIL import Image
import numpy as np

load_dotenv()

# ---------------- CONFIG ----------------
# High priority coins - scan every 30 minutes
HIGH_PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Alt coins - scan every 1 hour
ALT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "AAVEUSDT", "TRXUSDT", 
    "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LTCUSDT"
]

HIGH_PRIORITY_INTERVAL = 1800  # 30 minutes
ALT_INTERVAL = 3600  # 1 hour
SCAN_DELAY = 10  # 10 seconds between each scan (rate limit)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2-vl-72b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 70.0))

# Binance API - Get last 999 candles for better analysis
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=999"

# ---------------- Utils ----------------

def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return f"{p:.8f}" if abs(p) < 0.01 else (f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}")

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return [None] * len(data)
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    sma = sum(data[:period]) / period
    ema.append(sma)
    
    # Calculate EMA for rest
    for i in range(period, len(data)):
        ema_val = (data[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)
    
    # Pad with None for initial values
    return [None] * (period - 1) + ema

def detect_support_resistance(highs, lows, closes, lookback=20):
    """Detect support and resistance levels using price action"""
    levels = []
    
    for i in range(lookback, len(closes) - lookback):
        # Check for resistance (local high)
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            levels.append(("resistance", highs[i]))
        
        # Check for support (local low)
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            levels.append(("support", lows[i]))
    
    # Remove duplicates and sort
    unique_levels = {}
    for level_type, price in levels:
        # Group similar prices (within 0.5%)
        found = False
        for existing_price in list(unique_levels.keys()):
            if abs(price - existing_price) / existing_price < 0.005:
                found = True
                break
        if not found:
            unique_levels[price] = level_type
    
    return sorted([(level_type, price) for price, level_type in unique_levels.items()], 
                  key=lambda x: x[1])

def detect_candlestick_patterns(opens, highs, lows, closes, volumes):
    """Detect candlestick patterns for last few candles"""
    patterns = []
    
    if len(closes) < 5:
        return patterns
    
    # Get last 5 candles
    o, h, l, c = opens[-5:], highs[-5:], lows[-5:], closes[-5:]
    
    # Current candle (index -1)
    curr_body = abs(c[-1] - o[-1])
    curr_range = h[-1] - l[-1]
    prev_body = abs(c[-2] - o[-2])
    
    # Doji - small body, long wicks
    if curr_range > 0 and curr_body / curr_range < 0.1:
        patterns.append("Doji (Indecision)")
    
    # Hammer - small body at top, long lower wick
    if curr_range > 0:
        lower_wick = min(o[-1], c[-1]) - l[-1]
        upper_wick = h[-1] - max(o[-1], c[-1])
        if lower_wick > curr_body * 2 and upper_wick < curr_body * 0.3:
            patterns.append("Hammer (Bullish Reversal)")
    
    # Shooting Star - small body at bottom, long upper wick
    if curr_range > 0:
        lower_wick = min(o[-1], c[-1]) - l[-1]
        upper_wick = h[-1] - max(o[-1], c[-1])
        if upper_wick > curr_body * 2 and lower_wick < curr_body * 0.3:
            patterns.append("Shooting Star (Bearish Reversal)")
    
    # Bullish Engulfing
    if c[-2] < o[-2] and c[-1] > o[-1]:  # Prev red, curr green
        if c[-1] > o[-2] and o[-1] < c[-2]:
            patterns.append("Bullish Engulfing (Strong Buy)")
    
    # Bearish Engulfing
    if c[-2] > o[-2] and c[-1] < o[-1]:  # Prev green, curr red
        if c[-1] < o[-2] and o[-1] > c[-2]:
            patterns.append("Bearish Engulfing (Strong Sell)")
    
    # Morning Star (3 candle pattern)
    if len(c) >= 3:
        if c[-3] < o[-3] and abs(c[-2] - o[-2]) < prev_body * 0.3 and c[-1] > o[-1]:
            if c[-1] > (o[-3] + c[-3]) / 2:
                patterns.append("Morning Star (Bullish Reversal)")
    
    # Evening Star (3 candle pattern)
    if len(c) >= 3:
        if c[-3] > o[-3] and abs(c[-2] - o[-2]) < prev_body * 0.3 and c[-1] < o[-1]:
            if c[-1] < (o[-3] + c[-3]) / 2:
                patterns.append("Evening Star (Bearish Reversal)")
    
    # Three White Soldiers
    if len(c) >= 3:
        if all(c[i] > o[i] for i in range(-3, 0)):
            if c[-3] < c[-2] < c[-1] and o[-3] < o[-2] < o[-1]:
                patterns.append("Three White Soldiers (Strong Bullish)")
    
    # Three Black Crows
    if len(c) >= 3:
        if all(c[i] < o[i] for i in range(-3, 0)):
            if c[-3] > c[-2] > c[-1] and o[-3] > o[-2] > o[-1]:
                patterns.append("Three Black Crows (Strong Bearish)")
    
    return patterns

def detect_chart_patterns(highs, lows, closes):
    """Detect chart patterns like head & shoulders, triangles, etc."""
    patterns = []
    
    if len(closes) < 50:
        return patterns
    
    recent_highs = highs[-50:]
    recent_lows = lows[-50:]
    
    # Find significant peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(5, len(recent_highs) - 5):
        if recent_highs[i] == max(recent_highs[i-5:i+6]):
            peaks.append((i, recent_highs[i]))
        if recent_lows[i] == min(recent_lows[i-5:i+6]):
            troughs.append((i, recent_lows[i]))
    
    # Head and Shoulders (3 peaks, middle highest)
    if len(peaks) >= 3:
        last_peaks = peaks[-3:]
        if last_peaks[1][1] > last_peaks[0][1] and last_peaks[1][1] > last_peaks[2][1]:
            if abs(last_peaks[0][1] - last_peaks[2][1]) / last_peaks[0][1] < 0.03:
                patterns.append("Head & Shoulders (Bearish Reversal)")
    
    # Inverse Head and Shoulders
    if len(troughs) >= 3:
        last_troughs = troughs[-3:]
        if last_troughs[1][1] < last_troughs[0][1] and last_troughs[1][1] < last_troughs[2][1]:
            if abs(last_troughs[0][1] - last_troughs[2][1]) / last_troughs[0][1] < 0.03:
                patterns.append("Inverse Head & Shoulders (Bullish Reversal)")
    
    # Ascending Triangle (higher lows, flat resistance)
    if len(troughs) >= 3 and len(peaks) >= 2:
        last_troughs = troughs[-3:]
        last_peaks = peaks[-2:]
        if all(last_troughs[i][1] < last_troughs[i+1][1] for i in range(len(last_troughs)-1)):
            if abs(last_peaks[0][1] - last_peaks[1][1]) / last_peaks[0][1] < 0.02:
                patterns.append("Ascending Triangle (Bullish)")
    
    # Descending Triangle (lower highs, flat support)
    if len(peaks) >= 3 and len(troughs) >= 2:
        last_peaks = peaks[-3:]
        last_troughs = troughs[-2:]
        if all(last_peaks[i][1] > last_peaks[i+1][1] for i in range(len(last_peaks)-1)):
            if abs(last_troughs[0][1] - last_troughs[1][1]) / last_troughs[0][1] < 0.02:
                patterns.append("Descending Triangle (Bearish)")
    
    # Double Top
    if len(peaks) >= 2:
        last_two_peaks = peaks[-2:]
        if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
            patterns.append("Double Top (Bearish Reversal)")
    
    # Double Bottom
    if len(troughs) >= 2:
        last_two_troughs = troughs[-2:]
        if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
            patterns.append("Double Bottom (Bullish Reversal)")
    
    return patterns

# ---------------- Chart Generation ----------------

def generate_enhanced_chart(symbol, candles_1h, candles_4h=None, candles_1d=None):
    """Generate advanced candlestick chart with EMAs, S/R, patterns"""
    try:
        if not candles_1h or len(candles_1h) < 50:
            return generate_error_chart(symbol, "Insufficient Data")

        # Parse 1H candle data
        times, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        
        for candle in candles_1h:
            try:
                times.append(datetime.utcfromtimestamp(int(candle[0]) / 1000))
                opens.append(float(candle[1]))
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
                volumes.append(float(candle[5]))
            except:
                continue

        if len(closes) < 50:
            return generate_error_chart(symbol, "Not enough valid candles")

        # Calculate EMAs (9, 20, 50)
        ema9 = calculate_ema(closes, 9)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)

        # Detect patterns
        sr_levels = detect_support_resistance(highs, lows, closes)
        candle_patterns = detect_candlestick_patterns(opens, highs, lows, closes, volumes)
        chart_patterns = detect_chart_patterns(highs, lows, closes)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.4, 0.4], hspace=0.05)
        
        ax1 = fig.add_subplot(gs[0])  # Price chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Volume
        ax3 = fig.add_subplot(gs[2])  # Patterns info
        ax4 = fig.add_subplot(gs[3])  # Trend info

        # Plot candlesticks on ax1
        x = date2num(times)
        for i in range(len(x)):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            alpha = 0.9
            
            # Wick
            ax1.plot([x[i], x[i]], [lows[i], highs[i]], 
                    color=color, linewidth=0.8, alpha=alpha)
            # Body
            ax1.plot([x[i], x[i]], [opens[i], closes[i]], 
                    color=color, linewidth=3.5, solid_capstyle='round', alpha=alpha)

        # Plot EMAs
        valid_ema9 = [(x[i], ema9[i]) for i in range(len(ema9)) if ema9[i]]
        valid_ema20 = [(x[i], ema20[i]) for i in range(len(ema20)) if ema20[i]]
        valid_ema50 = [(x[i], ema50[i]) for i in range(len(ema50)) if ema50[i]]
        
        if valid_ema9:
            ax1.plot([v[0] for v in valid_ema9], [v[1] for v in valid_ema9], 
                    label='EMA 9', linewidth=1.5, color='#ff6b6b', alpha=0.8)
        if valid_ema20:
            ax1.plot([v[0] for v in valid_ema20], [v[1] for v in valid_ema20], 
                    label='EMA 20', linewidth=1.5, color='#4ecdc4', alpha=0.8)
        if valid_ema50:
            ax1.plot([v[0] for v in valid_ema50], [v[1] for v in valid_ema50], 
                    label='EMA 50', linewidth=1.5, color='#ffa726', alpha=0.8)

        # Plot Support/Resistance levels
        for level_type, price in sr_levels[-5:]:  # Show last 5 levels
            color = '#ff5252' if level_type == 'resistance' else '#69f0ae'
            linestyle = '--' if level_type == 'resistance' else '-.'
            ax1.axhline(y=price, color=color, linestyle=linestyle, 
                       linewidth=1.2, alpha=0.6, 
                       label=f'{level_type.title()}: {fmt_price(price)}')

        # Styling
        ax1.set_title(f"{symbol} - 1H Chart | Price Action Analysis", 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax1.set_facecolor('#fafafa')
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Volume chart
        vol_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                     for i in range(len(closes))]
        ax2.bar(x, volumes, color=vol_colors, alpha=0.6, width=0.0006)
        ax2.set_ylabel('Volume', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax2.set_facecolor('#fafafa')
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

        # Patterns info box
        ax3.axis('off')
        pattern_text = "🔍 CANDLESTICK PATTERNS:\n"
        if candle_patterns:
            pattern_text += "\n".join([f"• {p}" for p in candle_patterns[:3]])
        else:
            pattern_text += "• No significant patterns"
        
        if chart_patterns:
            pattern_text += "\n\n📊 CHART PATTERNS:\n"
            pattern_text += "\n".join([f"• {p}" for p in chart_patterns[:2]])
        
        ax3.text(0.05, 0.5, pattern_text, fontsize=8, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Trend info box
        ax4.axis('off')
        current_price = closes[-1]
        trend_text = f"💰 Current: {fmt_price(current_price)}"
        
        # Determine trend based on EMAs
        if ema9[-1] and ema20[-1] and ema50[-1]:
            if current_price > ema9[-1] > ema20[-1] > ema50[-1]:
                trend_text += " | 📈 STRONG UPTREND"
            elif current_price < ema9[-1] < ema20[-1] < ema50[-1]:
                trend_text += " | 📉 STRONG DOWNTREND"
            elif current_price > ema20[-1]:
                trend_text += " | 📊 UPTREND"
            elif current_price < ema20[-1]:
                trend_text += " | 📊 DOWNTREND"
            else:
                trend_text += " | ↔️ SIDEWAYS"
        
        ax4.text(0.05, 0.5, trend_text, fontsize=9, fontweight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()

    except Exception as e:
        print(f"❌ Chart generation error: {e}")
        traceback.print_exc()
        return generate_error_chart(symbol, str(e))

def generate_error_chart(symbol, error_msg):
    """Generate simple error chart"""
    fig = plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, f"{symbol}\n\n⚠️ {error_msg}", 
            ha='center', va='center', fontsize=16, color='red')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Prepare Raw Data for AI ----------------

def prepare_raw_data(candles_1h, candles_4h=None, candles_1d=None):
    """Prepare raw market data to send with chart"""
    data = {}
    
    # 1H data
    if candles_1h and len(candles_1h) >= 50:
        recent_1h = candles_1h[-50:]  # Last 50 candles
        data['1h'] = {
            'candles': [[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] 
                       for c in recent_1h],
            'timestamps': [int(c[0]) for c in recent_1h]
        }
    
    # 4H data (for multi-timeframe)
    if candles_4h and len(candles_4h) >= 20:
        recent_4h = candles_4h[-20:]
        data['4h'] = {
            'candles': [[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] 
                       for c in recent_4h],
            'timestamps': [int(c[0]) for c in recent_4h]
        }
    
    # 1D data (for trend context)
    if candles_1d and len(candles_1d) >= 10:
        recent_1d = candles_1d[-10:]
        data['1d'] = {
            'candles': [[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] 
                       for c in recent_1d],
            'timestamps': [int(c[0]) for c in recent_1d]
        }
    
    return json.dumps(data, indent=2)

# ---------------- Qwen Vision Analysis ----------------

async def analyze_with_qwen(session, image_bytes, symbol, raw_data, current_price):
    """Enhanced Qwen analysis with raw data"""
    if not OPENROUTER_API_KEY:
        return create_error_signal("OpenRouter API key not configured")

    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://crypto-scanner.local",
            "X-Title": "Crypto Chart Analyzer"
        }

        prompt = f"""🎯 ADVANCED CRYPTO ANALYSIS - {symbol}

Current Price: {fmt_price(current_price)}

📊 RAW MARKET DATA (OHLCV):
```json
{raw_data}
        payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "top_p": 0.9
    }

    async with session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90) as response:
        if response.status != 200:
            error_text = await response.text()
            print(f"❌ API error {response.status}: {error_text[:200]}")
            return create_error_signal(f"API error: {response.status}")

        result = await response.json()
        analysis_text = result['choices'][0]['message']['content']
        
        signal_data = parse_enhanced_analysis(analysis_text, current_price)
        signal_data['analysis'] = analysis_text
        
        usage = result.get('usage', {})
        print(f"   🤖 Tokens: {usage.get('total_tokens', 'N/A')}")
        
        return signal_data

except asyncio.TimeoutError:
    return create_error_signal("API timeout")
except Exception as e:
    print(f"❌ Analysis error: {e}")
    traceback.print_exc()
    return create_error_signal(str(e))
        ": f"Error: {reason}"
    }

def parse_enhanced_analysis(text, current_price):
    """Parse enhanced Qwen analysis"""
    try:
        import re
        text_upper = text.upper()
        
        # Detect action
        side = "none"
        if "**ACTION**: BUY" in text or "**ACTION**: BUY" in text_upper:
            side = "BUY"
        elif "**ACTION**: SELL" in text or "**ACTION**: SELL" in text_upper:
            side = "SELL"
        elif "ACTION: BUY" in text_upper:
            side = "BUY"
        elif "ACTION: SELL" in text_upper:
            side = "SELL"
        
        # Extract confidence
        confidence = 0
        conf_match = re.search(r'CONFIDENCE[:\s*]*(\d+)/10', text_upper)
        if conf_match:
            confidence = int(conf_match.group(1)) * 10
        else:
            conf_match = re.search(r'CONFIDENCE[:\s*]*(\d+)%', text_upper)
            if conf_match:
                confidence = int(conf_match.group(1))
        
        # Extract prices
        entry = current_price
        entry_match = re.search(r'ENTRY[^\d]*(\d+\.?\d*)', text)
        if entry_match:
            try:
                entry = float(entry_match.group(1))
            except:
                pass
        
        sl = entry * 0.98 if side == "BUY" else entry * 1.02
        sl_match = re.search(r'STOP LOSS[^\d]*(\d+\.?\d*)', text)
        if sl_match:
            try:
                sl = float(sl_match.group(1))
            except:
                pass
        
        tp1 = entry * 1.02 if side == "BUY" else entry * 0.98
        tp1_match = re.search(r'TAKE PROFIT[^\d]*1[^\d]*(\d+\.?\d*)', text)
        if tp1_match:
            try:
                tp1 = float(tp1_match.group(1))
            except:
                pass
        
        tp2 = entry * 1.04 if side == "BUY" else entry * 0.96
        tp2_match = re.search(r'TAKE PROFIT[^\d]*2[^\d]*(\d+\.?\d*)', text)
        if tp2_match:
            try:
                tp2 = float(tp2_match.group(1))
            except:
                pass
        
        # Calculate R:R
        rr = 0
        if side == "BUY" and (entry - sl) > 0:
            rr = (tp1 - entry) / (entry - sl)
        elif side == "SELL" and (sl - entry) > 0:
            rr = (entry - tp1) / (sl - entry)
        
        # Extract reasoning
        reason_match = re.search(r'REASONING[:\s*]*(.{50,300})', text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip()[:200] if reason_match else "See full analysis"
        
        return {
            "side": side,
            "confidence": confidence,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr": round(rr, 2),
            "reason": reason
        }
    
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return {
            "side": "none",
            "confidence": 0,
            "entry": current_price,
            "sl": 0,
            "tp1": 0,
            "tp2": 0,
            "rr": 0,
            "reason": "Failed to parse analysis"
        }

# ---------------- Fetch Data ----------------

async def fetch_json(session, url):
    """Fetch JSON from URL"""
    try:
        async with session.get(url, timeout=30) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception as e:
        print(f"   ⚠️ Fetch error: {e}")
        return None

async def fetch_multi_timeframe_data(session, symbol):
    """Fetch 1H, 4H, 1D candles for multi-timeframe analysis"""
    try:
        # Fetch all timeframes concurrently
        tasks = [
            fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='1h')),
            fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='4h')),
            fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='1d'))
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        candles_1h = results[0] if not isinstance(results[0], Exception) else None
        candles_4h = results[1] if not isinstance(results[1], Exception) else None
        candles_1d = results[2] if not isinstance(results[2], Exception) else None
        
        return candles_1h, candles_4h, candles_1d
    
    except Exception as e:
        print(f"   ⚠️ Multi-TF fetch error: {e}")
        return None, None, None

# ---------------- Telegram ----------------

async def send_text(session, text):
    """Send text to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": text,
            "parse_mode": "HTML"
        }) as resp:
            if resp.status != 200:
                print(f"⚠️ Telegram error: {resp.status}")
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")

async def send_photo_bytes(session, caption, image_bytes):
    """Send photo to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        data = aiohttp.FormData()
        data.add_field("chat_id", TELEGRAM_CHAT_ID)
        data.add_field("caption", caption[:1024])
        data.add_field("photo", image_bytes, filename="chart.png", content_type="image/png")
        
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"⚠️ Telegram photo error {resp.status}: {error_text[:100]}")
    except Exception as e:
        print(f"⚠️ Telegram photo exception: {e}")

# ---------------- Scanning Logic ----------------

async def scan_symbol(session, symbol, scan_type="normal"):
    """Scan individual symbol with enhanced analysis"""
    try:
        print(f"   📊 Fetching multi-timeframe data...")
        
        # Fetch 1H, 4H, 1D data
        candles_1h, candles_4h, candles_1d = await fetch_multi_timeframe_data(session, symbol)
        
        if not candles_1h or len(candles_1h) < 50:
            print(f"   ❌ Insufficient 1H data")
            return None
        
        # Get current price
        current_price = float(candles_1h[-1][4])
        print(f"   💰 Price: {fmt_price(current_price)}")
        
        # Generate enhanced chart
        print(f"   📈 Generating chart with EMAs & patterns...")
        chart_bytes = generate_enhanced_chart(symbol, candles_1h, candles_4h, candles_1d)
        
        # Prepare raw data for AI
        print(f"   📋 Preparing raw market data...")
        raw_data = prepare_raw_data(candles_1h, candles_4h, candles_1d)
        
        # Analyze with Qwen
        print(f"   🤖 AI Analysis (Multi-TF)...")
        signal = await analyze_with_qwen(session, chart_bytes, symbol, raw_data, current_price)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "signal": signal,
            "chart": chart_bytes
        }
    
    except Exception as e:
        print(f"   ❌ Scan error: {e}")
        traceback.print_exc()
        return None

# ---------------- Main Scanner ----------------

async def main_scanner_loop():
    """Main enhanced scanner with dual intervals"""
    
    async with aiohttp.ClientSession() as session:
        startup_msg = f"""🚀 <b>QWEN VL CRYPTO SCANNER V2.0</b>

📊 <b>Configuration:</b>
├─ High Priority: {', '.join(HIGH_PRIORITY_SYMBOLS)}
│  └─ Scan: Every 30 minutes
├─ Alt Coins: {', '.join(ALT_SYMBOLS)}
│  └─ Scan: Every 1 hour
├─ Data: Last 999 candles per TF
├─ Timeframes: 1H, 4H, 1D (Multi-TF)
└─ Indicators: EMA 9/20/50 only

🎯 <b>Features:</b>
├─ ✅ Price Action Analysis
├─ ✅ Support/Resistance Detection
├─ ✅ Candlestick Patterns (All types)
├─ ✅ Chart Patterns Recognition
├─ ✅ Trendline Analysis
├─ ✅ Multi-Timeframe Confluence
└─ ✅ Raw Data + Chart to AI

⚙️ <b>Settings:</b>
├─ Model: {OPENROUTER_MODEL.split('/')[-1]}
├─ Rate Limit: {SCAN_DELAY}s between scans
└─ Min Confidence: {SIGNAL_CONF_THRESHOLD}%

🟢 Scanner Active - Starting first scan..."""
        
        print(startup_msg.replace('<b>', '').replace('</b>', ''))
        await send_text(session, startup_msg)
        
        iteration = 0
        last_high_priority_scan = 0
        last_alt_scan = 0
        
        while True:
            iteration += 1
            current_time = asyncio.get_event_loop().time()
            scan_start = datetime.utcnow()
            
            # Determine what to scan
            scan_high_priority = (current_time - last_high_priority_scan) >= HIGH_PRIORITY_INTERVAL
            scan_alts = (current_time - last_alt_scan) >= ALT_INTERVAL
            
            symbols_to_scan = []
            scan_type_msg = ""
            
            if scan_high_priority:
                symbols_to_scan.extend(HIGH_PRIORITY_SYMBOLS)
                scan_type_msg += "🔥 High Priority (30min) "
                last_high_priority_scan = current_time
            
            if scan_alts:
                symbols_to_scan.extend(ALT_SYMBOLS)
                scan_type_msg += "📊 Alt Coins (1H) "
                last_alt_scan = current_time
            
            if not symbols_to_scan:
                # Nothing to scan yet, wait a bit
                await asyncio.sleep(60)
                continue
            
            print(f"\n{'='*80}")
            print(f"🔍 SCAN #{iteration} @ {scan_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"📋 {scan_type_msg}")
            print(f"{'='*80}")
            
            signals_found = 0
            scan_results = []
            
            for idx, symbol in enumerate(symbols_to_scan, 1):
                try:
                    print(f"\n[{idx}/{len(symbols_to_scan)}] 🔍 {symbol}...")
                    
                    result = await scan_symbol(session, symbol)
                    
                    if not result:
                        continue
                    
                    signal = result['signal']
                    
                    # Check if valid signal
                    if signal['side'] != 'none' and signal['confidence'] >= SIGNAL_CONF_THRESHOLD:
                        signals_found += 1
                        
                        # Format signal message
                        emoji = "🟢" if signal['side'] == "BUY" else "🔴"
                        
                        msg = f"""{emoji} <b>{symbol} {signal['side']} SIGNAL</b>

💰 <b>Price:</b> {fmt_price(result['price'])}
📊 <b>Confidence:</b> {signal['confidence']}%

🎯 <b>Trade Setup:</b>
├─ Entry: {fmt_price(signal['entry'])}
├─ Stop Loss: {fmt_price(signal['sl'])}
├─ TP1: {fmt_price(signal['tp1'])}
├─ TP2: {fmt_price(signal['tp2'])}
└─ R:R Ratio: 1:{signal['rr']}

💡 <b>Reasoning:</b>
{signal.get('reason', 'See detailed analysis below')[:150]}

⚠️ <b>Risk Management:</b>
Risk only 1-2% of capital per trade!

🤖 Qwen 2.5 VL 72B | Multi-TF Analysis"""
                        
                        # Send chart with signal
                        await send_photo_bytes(session, msg, result['chart'])
                        
                        print(f"   ✅ SIGNAL: {signal['side']} | Conf: {signal['confidence']}% | R:R 1:{signal['rr']}")
                        
                        scan_results.append({
                            "symbol": symbol,
                            "side": signal['side'],
                            "confidence": signal['confidence'],
                            "rr": signal['rr']
                        })
                        
                        # Send detailed analysis as separate message
                        if len(signal.get('analysis', '')) > 100:
                            analysis_msg = f"📝 <b>Detailed Analysis - {symbol}</b>\n\n{signal['analysis'][:3000]}"
                            await send_text(session, analysis_msg)
                    
                    else:
                        print(f"   📉 No signal | Conf: {signal['confidence']}% | Action: {signal['side']}")
                        if signal.get('reason'):
                            print(f"      💭 {signal['reason'][:100]}")
                    
                    # Rate limit delay
                    if idx < len(symbols_to_scan):
                        print(f"   ⏳ Rate limit delay ({SCAN_DELAY}s)...")
                        await asyncio.sleep(SCAN_DELAY)
                
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    traceback.print_exc()
                    continue
            
            # Scan summary
            scan_duration = (datetime.utcnow() - scan_start).total_seconds()
            
            summary = f"""📊 <b>Scan #{iteration} Complete</b>

✅ Analyzed: {len(symbols_to_scan)} symbols
🎯 Signals Found: {signals_found}
⏱️ Duration: {scan_duration:.1f}s

{'─'*40}"""
            
            if signals_found > 0:
                summary += "\n\n🚀 <b>Active Signals:</b>\n"
                for r in scan_results:
                    emoji = "🟢" if r['side'] == "BUY" else "🔴"
                    summary += f"{emoji} {r['symbol']}: {r['side']} ({r['confidence']}%) R:R 1:{r['rr']}\n"
            
            # Next scan info
            next_high = HIGH_PRIORITY_INTERVAL - (asyncio.get_event_loop().time() - last_high_priority_scan)
            next_alt = ALT_INTERVAL - (asyncio.get_event_loop().time() - last_alt_scan)
            
            summary += f"\n\n⏰ <b>Next Scans:</b>"
            summary += f"\n├─ High Priority: {int(next_high//60)}m {int(next_high%60)}s"
            summary += f"\n└─ Alt Coins: {int(next_alt//60)}m {int(next_alt%60)}s"
            
            print(f"\n{summary.replace('<b>', '').replace('</b>', '')}")
            await send_text(session, summary)
            
            # Wait until next scan needed
            wait_time = min(next_high, next_alt)
            if wait_time > 0:
                print(f"\n💤 Sleeping for {int(wait_time)}s...")
                await asyncio.sleep(max(wait_time, 30))

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    try:
        print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🤖 QWEN VL CRYPTO SCANNER V2.0                       ║
║     Advanced Price Action & Pattern Recognition          ║
║                                                           ║
║     Features:                                            ║
║     ✅ Multi-Timeframe Analysis (1H/4H/1D)               ║
║     ✅ EMA 9/20/50 Trend Detection                       ║
║     ✅ Support/Resistance Levels                         ║
║     ✅ All Candlestick Patterns                          ║
║     ✅ Chart Patterns Recognition                        ║
║     ✅ Raw Data + Visual Analysis                        ║
║     ✅ Dual Interval Scanning                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
        asyncio.run(main_scanner_loop())
        
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        
