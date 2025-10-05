#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v5.0 - Price Action Master Edition
# Major improvements: Advanced candlestick patterns, chart patterns, trendlines, S/R zones
# Pure price action + EMA9/EMA20 confluence for highest accuracy

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
from matplotlib.patches import Rectangle
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
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"
MIN_CANDLES_REQUIRED = 720

# ---------------- Utils ----------------
def fmt_price(p):
    try:
        p = float(p)
    except:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"

def ema(values, period):
    if not values or len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema_vals.append(sma)
    prev = sma
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        ema_vals.append(prev)
    return ema_vals

def calculate_emas(closes):
    return {
        'ema_9': ema(closes, 9)[-1] if len(closes) >= 9 else None,
        'ema_20': ema(closes, 20)[-1] if len(closes) >= 20 else None,
        'ema_9_series': ema(closes, 9),
        'ema_20_series': ema(closes, 20)
    }

def normalize_klines(raw_klines):
    out = []
    for row in raw_klines or []:
        try:
            if len(row) >= 6:
                out.append({
                    "open": float(row[1]), "high": float(row[2]), 
                    "low": float(row[3]), "close": float(row[4]),
                    "volume": float(row[5]), "ts": int(row[0])
                })
        except:
            continue
    return out

# ---------------- ADVANCED CANDLESTICK PATTERN DETECTION ----------------
def detect_candlestick_patterns(candles):
    """Detect bullish/bearish candlestick patterns"""
    if len(candles) < 3:
        return []
    
    patterns = []
    
    for i in range(2, len(candles)):
        c0, c1, c2 = candles[i-2], candles[i-1], candles[i]
        o0, h0, l0, cl0 = c0['open'], c0['high'], c0['low'], c0['close']
        o1, h1, l1, cl1 = c1['open'], c1['high'], c1['low'], c1['close']
        o2, h2, l2, cl2 = c2['open'], c2['high'], c2['low'], c2['close']
        
        body1 = abs(cl1 - o1)
        body2 = abs(cl2 - o2)
        range1 = h1 - l1
        range2 = h2 - l2
        
        # Bullish Patterns
        
        # Hammer / Inverted Hammer
        if body2 > 0.001:
            lower_shadow = min(o2, cl2) - l2
            upper_shadow = h2 - max(o2, cl2)
            if lower_shadow > body2 * 2 and upper_shadow < body2 * 0.3 and cl2 > o2:
                patterns.append({"type": "Hammer", "sentiment": "bullish", "strength": 8, "index": i})
            elif upper_shadow > body2 * 2 and lower_shadow < body2 * 0.3 and cl2 > o2:
                patterns.append({"type": "Inverted Hammer", "sentiment": "bullish", "strength": 7, "index": i})
        
        # Bullish Engulfing
        if cl0 < o0 and cl2 > o2 and cl2 >= o0 and o2 <= cl0:
            patterns.append({"type": "Bullish Engulfing", "sentiment": "bullish", "strength": 9, "index": i})
        
        # Morning Star
        if len(candles) >= i+1:
            if cl0 < o0 and body1 < body2 * 0.3 and cl2 > o2 and cl2 > (o0 + cl0)/2:
                patterns.append({"type": "Morning Star", "sentiment": "bullish", "strength": 10, "index": i})
        
        # Piercing Pattern
        if cl0 < o0 and cl2 > o2 and o2 < cl0 and cl2 > (o0 + cl0)/2 and cl2 < o0:
            patterns.append({"type": "Piercing Pattern", "sentiment": "bullish", "strength": 8, "index": i})
        
        # Three White Soldiers
        if i >= 3:
            c3 = candles[i-3]
            if (cl2 > o2 and cl1 > o1 and c3['close'] > c3['open'] and
                cl2 > cl1 > c3['close'] and body2 > 0 and body1 > 0):
                patterns.append({"type": "Three White Soldiers", "sentiment": "bullish", "strength": 10, "index": i})
        
        # Bearish Patterns
        
        # Shooting Star / Hanging Man
        if body2 > 0.001:
            lower_shadow = min(o2, cl2) - l2
            upper_shadow = h2 - max(o2, cl2)
            if upper_shadow > body2 * 2 and lower_shadow < body2 * 0.3 and cl2 < o2:
                patterns.append({"type": "Shooting Star", "sentiment": "bearish", "strength": 8, "index": i})
            elif lower_shadow > body2 * 2 and upper_shadow < body2 * 0.3 and cl2 < o2:
                patterns.append({"type": "Hanging Man", "sentiment": "bearish", "strength": 7, "index": i})
        
        # Bearish Engulfing
        if cl0 > o0 and cl2 < o2 and cl2 <= o0 and o2 >= cl0:
            patterns.append({"type": "Bearish Engulfing", "sentiment": "bearish", "strength": 9, "index": i})
        
        # Evening Star
        if cl0 > o0 and body1 < body2 * 0.3 and cl2 < o2 and cl2 < (o0 + cl0)/2:
            patterns.append({"type": "Evening Star", "sentiment": "bearish", "strength": 10, "index": i})
        
        # Dark Cloud Cover
        if cl0 > o0 and cl2 < o2 and o2 > cl0 and cl2 < (o0 + cl0)/2 and cl2 > o0:
            patterns.append({"type": "Dark Cloud Cover", "sentiment": "bearish", "strength": 8, "index": i})
        
        # Three Black Crows
        if i >= 3:
            c3 = candles[i-3]
            if (cl2 < o2 and cl1 < o1 and c3['close'] < c3['open'] and
                cl2 < cl1 < c3['close'] and body2 > 0 and body1 > 0):
                patterns.append({"type": "Three Black Crows", "sentiment": "bearish", "strength": 10, "index": i})
        
        # Doji variations
        if body2 < range2 * 0.1 and range2 > 0:
            patterns.append({"type": "Doji", "sentiment": "neutral", "strength": 5, "index": i})
    
    return patterns

# ---------------- CHART PATTERN DETECTION ----------------
def detect_chart_patterns(highs, lows, closes, lookback=50):
    """Detect chart patterns like triangles, H&S, double tops/bottoms"""
    patterns = []
    n = min(len(highs), lookback)
    if n < 20:
        return patterns
    
    recent_highs = highs[-n:]
    recent_lows = lows[-n:]
    recent_closes = closes[-n:]
    
    # Find significant swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(5, n-5):
        if recent_highs[i] == max(recent_highs[i-5:i+5]):
            swing_highs.append((i, recent_highs[i]))
        if recent_lows[i] == min(recent_lows[i-5:i+5]):
            swing_lows.append((i, recent_lows[i]))
    
    # Double Top/Bottom detection
    if len(swing_highs) >= 2:
        last_two_highs = swing_highs[-2:]
        if abs(last_two_highs[0][1] - last_two_highs[1][1]) / last_two_highs[0][1] < 0.02:
            patterns.append({"type": "Double Top", "sentiment": "bearish", "strength": 8})
    
    if len(swing_lows) >= 2:
        last_two_lows = swing_lows[-2:]
        if abs(last_two_lows[0][1] - last_two_lows[1][1]) / last_two_lows[0][1] < 0.02:
            patterns.append({"type": "Double Bottom", "sentiment": "bullish", "strength": 8})
    
    # Triangle patterns (simplified)
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        high_trend = (swing_highs[-1][1] - swing_highs[-3][1]) / swing_highs[-3][1]
        low_trend = (swing_lows[-1][1] - swing_lows[-3][1]) / swing_lows[-3][1]
        
        if abs(high_trend) < 0.01 and low_trend > 0.02:
            patterns.append({"type": "Ascending Triangle", "sentiment": "bullish", "strength": 7})
        elif high_trend < -0.02 and abs(low_trend) < 0.01:
            patterns.append({"type": "Descending Triangle", "sentiment": "bearish", "strength": 7})
        elif abs(high_trend) < 0.015 and abs(low_trend) < 0.015:
            patterns.append({"type": "Symmetrical Triangle", "sentiment": "neutral", "strength": 6})
    
    # Head and Shoulders (simplified detection)
    if len(swing_highs) >= 3:
        h = swing_highs[-3:]
        if h[1][1] > h[0][1] and h[1][1] > h[2][1] and abs(h[0][1] - h[2][1]) / h[0][1] < 0.03:
            patterns.append({"type": "Head and Shoulders", "sentiment": "bearish", "strength": 9})
    
    return patterns

# ---------------- SUPPORT/RESISTANCE ZONES ----------------
def calculate_sr_zones(closes, highs, lows, lookback=100):
    """Calculate support and resistance zones with strength"""
    n = min(len(closes), lookback)
    prices = list(closes[-n:]) + list(highs[-n:]) + list(lows[-n:])
    
    zones = []
    tolerance = 0.003  # 0.3% clustering
    
    for p in prices:
        if p is None or p == 0:
            continue
        found = False
        for zone in zones:
            if abs((zone['price'] - p) / p) < tolerance:
                zone['touches'] += 1
                zone['price'] = (zone['price'] * (zone['touches']-1) + p) / zone['touches']
                found = True
                break
        if not found:
            zones.append({'price': p, 'touches': 1})
    
    # Sort by touches and get top zones
    zones.sort(key=lambda x: -x['touches'])
    current = closes[-1]
    
    support_zones = [z for z in zones if z['price'] < current][:3]
    resistance_zones = [z for z in zones if z['price'] > current][:3]
    
    return {
        'support': [z['price'] for z in support_zones],
        'resistance': [z['price'] for z in resistance_zones],
        'support_strength': [z['touches'] for z in support_zones],
        'resistance_strength': [z['touches'] for z in resistance_zones]
    }

# ---------------- TRENDLINE DETECTION ----------------
def detect_trendlines(highs, lows, closes, lookback=100):
    """Detect major trendlines"""
    n = min(len(closes), lookback)
    if n < 20:
        return {'support_line': None, 'resistance_line': None}
    
    indices = np.arange(n)
    recent_highs = np.array(highs[-n:])
    recent_lows = np.array(lows[-n:])
    
    # Fit linear regression for support (lows) and resistance (highs)
    try:
        # Support trendline
        support_slope, support_intercept = np.polyfit(indices, recent_lows, 1)
        support_line = support_slope * indices + support_intercept
        
        # Resistance trendline
        resistance_slope, resistance_intercept = np.polyfit(indices, recent_highs, 1)
        resistance_line = resistance_slope * indices + resistance_intercept
        
        return {
            'support_line': (support_slope, support_intercept),
            'resistance_line': (resistance_slope, resistance_intercept),
            'support_trend': 'bullish' if support_slope > 0 else 'bearish',
            'resistance_trend': 'bullish' if resistance_slope > 0 else 'bearish'
        }
    except:
        return {'support_line': None, 'resistance_line': None}

# ---------------- ENHANCED TRADE ANALYSIS ----------------
def analyze_trade_logic(raw_candles, rr_min=1.8):
    """Enhanced trade analysis with price action, patterns, S/R, trendlines"""
    try:
        if not raw_candles:
            return {"side": "none", "confidence": 0, "reason": "no data"}
        
        candles = normalize_klines(raw_candles) if not isinstance(raw_candles[0], dict) else raw_candles
        
        if len(candles) < MIN_CANDLES_REQUIRED:
            return {"side": "none", "confidence": 0, "reason": f"need {MIN_CANDLES_REQUIRED} candles"}
        
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        current_price = closes[-1]
        
        # Calculate EMAs
        emas = calculate_emas(closes)
        ema9, ema20 = emas['ema_9'], emas['ema_20']
        
        if not ema9 or not ema20:
            return {"side": "none", "confidence": 0, "reason": "insufficient EMA data"}
        
        # Detect patterns
        candle_patterns = detect_candlestick_patterns(candles[-50:])
        chart_patterns = detect_chart_patterns(highs, lows, closes, lookback=100)
        sr_zones = calculate_sr_zones(closes, highs, lows, lookback=100)
        trendlines = detect_trendlines(highs, lows, closes, lookback=100)
        
        # Score calculation
        score = 0
        reasons = []
        detected_patterns = []
        
        # EMA signals
        if current_price > ema9:
            score += 2
            reasons.append("Price > EMA9")
        else:
            score -= 1
        
        if ema9 > ema20:
            score += 3
            reasons.append("EMA9 > EMA20 (bullish)")
        else:
            score -= 3
            reasons.append("EMA9 < EMA20 (bearish)")
        
        # Candlestick pattern scoring
        bullish_patterns = [p for p in candle_patterns if p['sentiment'] == 'bullish']
        bearish_patterns = [p for p in candle_patterns if p['sentiment'] == 'bearish']
        
        for p in bullish_patterns[-3:]:
            score += p['strength'] / 2
            detected_patterns.append(p['type'])
        
        for p in bearish_patterns[-3:]:
            score -= p['strength'] / 2
            detected_patterns.append(p['type'])
        
        # Chart pattern scoring
        for p in chart_patterns:
            if p['sentiment'] == 'bullish':
                score += p['strength'] / 2
                detected_patterns.append(p['type'])
            elif p['sentiment'] == 'bearish':
                score -= p['strength'] / 2
                detected_patterns.append(p['type'])
        
        # Support/Resistance proximity
        supports = sr_zones['support']
        resistances = sr_zones['resistance']
        
        near_support = any(abs(current_price - s) / current_price < 0.01 for s in supports[:2]) if supports else False
        near_resistance = any(abs(current_price - r) / current_price < 0.01 for r in resistances[:2]) if resistances else False
        
        if near_support and score > 0:
            score += 3
            reasons.append("Near strong support")
        if near_resistance and score < 0:
            score += 3
            reasons.append("Near strong resistance")
        
        # Trendline alignment
        if trendlines.get('support_trend') == 'bullish' and score > 0:
            score += 2
            reasons.append("Bullish trendline")
        if trendlines.get('resistance_trend') == 'bearish' and score < 0:
            score += 2
            reasons.append("Bearish trendline")
        
        # Base confidence
        base_conf = 50 + abs(score) * 3
        
        # Generate signals
        if score >= 5:  # Strong bullish
            support = supports[0] if supports else current_price * 0.98
            resistance = resistances[0] if resistances else current_price * 1.05
            
            entry = float(current_price)
            sl = float(support) * 0.996
            tp = float(resistance)
            
            if (entry - sl) > 0:
                rr = (tp - entry) / (entry - sl)
                if rr >= rr_min:
                    confidence = min(95, int(base_conf))
                    return {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": confidence,
                        "reason": "; ".join(reasons[:4]),
                        "score": round(score, 2),
                        "rr": round(rr, 2),
                        "patterns": detected_patterns[:3],
                        "indicators": {"ema_9": ema9, "ema_20": ema20},
                        "sr_zones": sr_zones,
                        "trendlines": trendlines
                    }
        
        elif score <= -5:  # Strong bearish
            support = supports[0] if supports else current_price * 0.95
            resistance = resistances[0] if resistances else current_price * 1.02
            
            entry = float(current_price)
            sl = float(resistance) * 1.004
            tp = float(support)
            
            if (sl - entry) > 0:
                rr = (entry - tp) / (sl - entry)
                if rr >= rr_min:
                    confidence = min(95, int(base_conf))
                    return {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": confidence,
                        "reason": "; ".join(reasons[:4]),
                        "score": round(score, 2),
                        "rr": round(rr, 2),
                        "patterns": detected_patterns[:3],
                        "indicators": {"ema_9": ema9, "ema_20": ema20},
                        "sr_zones": sr_zones,
                        "trendlines": trendlines
                    }
        
        return {
            "side": "none",
            "confidence": int(base_conf),
            "reason": "; ".join(reasons[:3]) if reasons else "no clear setup",
            "score": round(score, 2),
            "patterns": detected_patterns[:3],
            "indicators": {"ema_9": ema9, "ema_20": ema20}
        }
    
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        return {"side": "none", "confidence": 0, "reason": f"error: {e}"}

# ---------------- ENHANCED CHART ----------------
def plot_signal_chart(symbol, raw_candles, signal):
    """Enhanced chart with patterns, S/R zones, trendlines"""
    candles = normalize_klines(raw_candles)
    if not candles or len(candles) < 50:
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6,2))
        plt.text(0.5,0.5,"Insufficient data",ha='center')
        plt.axis('off')
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name
    
    # Prepare data
    dates = [datetime.utcfromtimestamp(c['ts']/1000) for c in candles]
    closes = [c['close'] for c in candles]
    opens = [c['open'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    
    x = date2num(dates)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [4, 1]})
    fig.patch.set_facecolor('#0a0e27')
    ax1.set_facecolor('#0f1419')
    ax2.set_facecolor('#0f1419')
    
    # Candlesticks
    for i, (xi, o, h, l, c) in enumerate(zip(x, opens, highs, lows, closes)):
        color = '#26a69a' if c >= o else '#ef5350'
        ax1.plot([xi, xi], [l, h], color=color, linewidth=0.6, alpha=0.8)
        ax1.plot([xi, xi], [o, c], color=color, linewidth=2.5, solid_capstyle='round')
    
    # EMAs
    try:
        ema9_series = ema(closes, 9)
        ema20_series = ema(closes, 20)
        if len(ema9_series) == len(x):
            ax1.plot(x, ema9_series, label='EMA 9', color='#42a5f5', linewidth=1.5, alpha=0.9)
        if len(ema20_series) == len(x):
            ax1.plot(x, ema20_series, label='EMA 20', color='#ffa726', linewidth=1.5, alpha=0.9)
    except:
        pass
    
    # Support/Resistance zones
    sr_zones = signal.get('sr_zones', {})
    for i, sup in enumerate(sr_zones.get('support', [])[:2]):
        strength = sr_zones.get('support_strength', [0])[i] if i < len(sr_zones.get('support_strength', [])) else 1
        alpha = min(0.3, 0.1 + strength * 0.02)
        ax1.axhline(sup, color='#4caf50', linestyle='--', linewidth=1.5, alpha=alpha, label=f'Support {fmt_price(sup)}')
        ax1.axhspan(sup*0.998, sup*1.002, alpha=alpha*0.3, color='#4caf50')
    
    for i, res in enumerate(sr_zones.get('resistance', [])[:2]):
        strength = sr_zones.get('resistance_strength', [0])[i] if i < len(sr_zones.get('resistance_strength', [])) else 1
        alpha = min(0.3, 0.1 + strength * 0.02)
        ax1.axhline(res, color='#f44336', linestyle='--', linewidth=1.5, alpha=alpha, label=f'Resistance {fmt_price(res)}')
        ax1.axhspan(res*0.998, res*1.002, alpha=alpha*0.3, color='#f44336')
    
    # Trendlines
    trendlines = signal.get('trendlines', {})
    if trendlines.get('support_line'):
        slope, intercept = trendlines['support_line']
        tl = slope * np.arange(len(x)) + intercept
        ax1.plot(x, tl, color='#66bb6a', linestyle=':', linewidth=2, alpha=0.6, label='Support Trendline')
    
    if trendlines.get('resistance_line'):
        slope, intercept = trendlines['resistance_line']
        tl = slope * np.arange(len(x)) + intercept
        ax1.plot(x, tl, color='#ef5350', linestyle=':', linewidth=2, alpha=0.6, label='Resistance Trendline')
    
    # Entry, SL, TP
    if signal.get('entry'):
        ax1.axhline(signal['entry'], color='#ffeb3b', linestyle='-', linewidth=2, label=f"Entry {fmt_price(signal['entry'])}")
    if signal.get('sl'):
        ax1.axhline(signal['sl'], color='#ff5252', linestyle='-', linewidth=2, label=f"Stop Loss {fmt_price(signal['sl'])}")
    if signal.get('tp'):
        ax1.axhline(signal['tp'], color='#69f0ae', linestyle='-', linewidth=2, label=f"Take Profit {fmt_price(signal['tp'])}")
    
    # Styling
    ax1.set_title(f'{symbol} - Price Action Analysis', color='white', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, facecolor='#1a1f2e', edgecolor='#2a2f3e')
    ax1.grid(True, alpha=0.15, color='#2a2f3e')
    ax1.tick_params(colors='white')
    
    # Volume
    volumes = [c['volume'] for c in candles]
    colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' for i in range(len(closes))]
    ax2.bar(x, volumes, color=colors, alpha=0.6, width=0.0008)
    ax2.set_title('Volume', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='#2a2f3e')
    
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight', facecolor='#0a0e27')
    plt.close(fig)
    return tmp.name

# ---------------- ENHANCED AI ANALYSIS ----------------
async def ask_openai_for_signals(symbol, signal_data):
    """Enhanced AI analysis with pattern recognition"""
    if not client:
        return None
    
    patterns_str = ", ".join(signal_data.get('patterns', [])[:3]) if signal_data.get('patterns') else "None"
    
    market_summary = {
        "symbol": symbol,
        "current_price": signal_data.get("entry", 0),
        "score": signal_data.get("score", 0),
        "ema_9": signal_data.get("indicators", {}).get("ema_9"),
        "ema_20": signal_data.get("indicators", {}).get("ema_20"),
        "detected_patterns": patterns_str,
        "proposed_trade": {
            "side": signal_data.get("side"),
            "confidence": signal_data.get("confidence"),
            "rr": signal_data.get("rr"),
            "reason": signal_data.get("reason")
        }
    }
    
    system_prompt = (
        "You are an elite crypto trader with 10+ years of experience in price action trading. "
        "Analyze the provided market data including candlestick patterns, chart patterns, EMAs, "
        "support/resistance zones, and trendlines. Either CONFIRM or REJECT the proposed trade. "
        "Provide: VERDICT (CONFIRM/REJECT), CONFIDENCE (0-100%), and a concise 2-sentence REASON."
    )
    
    user_prompt = f"""Market Analysis for {symbol}:
    
Price: {market_summary['current_price']}
Technical Score: {market_summary['score']}
EMA 9: {market_summary['ema_9']}
EMA 20: {market_summary['ema_20']}

Detected Patterns: {market_summary['detected_patterns']}

Proposed Trade:
- Side: {market_summary['proposed_trade']['side']}
- Confidence: {market_summary['proposed_trade']['confidence']}%
- Risk/Reward: 1:{market_summary['proposed_trade']['rr']}
- Reason: {market_summary['proposed_trade']['reason']}

Respond in format:
VERDICT: [CONFIRM/REJECT]
CONFIDENCE: [0-100]%
REASON: [Your 2-sentence analysis]"""

    try:
        loop = asyncio.get_running_loop()
        def call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
        resp = await loop.run_in_executor(None, call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

# ---------------- Fetch ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=60) as r:
            if r.status != 200:
                text = await r.text()
                print(f"Fetch {url} -> {r.status}: {text[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print(f"Fetch error {url}: {e}")
        return None

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        print(f"Telegram text error: {e}")

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption)
        try:
            os.unlink(path)
        except:
            pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", TELEGRAM_CHAT_ID)
            data.add_field("caption", caption, content_type="text/html")
            data.add_field("parse_mode", "HTML")
            data.add_field("photo", f)
            await session.post(url, data=data)
    except Exception as e:
        print(f"Telegram photo error: {e}")
    finally:
        try:
            os.unlink(path)
        except:
            pass

# ---------------- MAIN ENHANCED LOOP ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup = (
            f"🚀 <b>Price Action Master Bot v5.0 Started!</b>\n\n"
            f"📊 Symbols: {len(SYMBOLS)}\n"
            f"⏱ Timeframe: 1H\n"
            f"📈 Candles: {MIN_CANDLES_REQUIRED}\n"
            f"🔄 Poll Interval: {POLL_INTERVAL}s\n"
            f"🎯 Min Confidence: {int(SIGNAL_CONF_THRESHOLD)}%\n\n"
            f"✨ Features:\n"
            f"• EMA 9 & 20\n"
            f"• 15+ Candlestick Patterns\n"
            f"• Chart Patterns (H&S, Triangles, Double Top/Bottom)\n"
            f"• Dynamic S/R Zones\n"
            f"• Trendline Detection\n"
            f"• AI-Enhanced Validation"
        )
        print(startup)
        await send_text(session, startup)

        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"🔍 ITERATION {iteration} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*70}")

            signals_found = 0
            analysis_summary = []

            for sym in SYMBOLS:
                try:
                    print(f"\n📊 Analyzing {sym}...")

                    # Fetch candle data
                    candles_raw = await fetch_json(session, CANDLE_URL.format(symbol=sym))

                    if not candles_raw:
                        print(f"❌ Failed to fetch data for {sym}")
                        continue

                    # Analyze with price action
                    local_signal = analyze_trade_logic(candles_raw)

                    analysis_summary.append({
                        "symbol": sym,
                        "confidence": local_signal.get("confidence", 0),
                        "side": local_signal.get("side", "none"),
                        "patterns": local_signal.get("patterns", [])
                    })

                    if local_signal.get("side") != "none" and local_signal.get("confidence", 0) >= SIGNAL_CONF_THRESHOLD:
                        # Ask AI for validation
                        ai_response = await ask_openai_for_signals(sym, local_signal)
                        
                        ai_boost = 0
                        ai_verdict = "PENDING"
                        
                        if ai_response:
                            if "CONFIRM" in ai_response.upper():
                                ai_boost = 8
                                ai_verdict = "✅ CONFIRMED"
                                print(f"✅ AI confirmed signal for {sym}")
                            elif "REJECT" in ai_response.upper():
                                ai_boost = -20
                                ai_verdict = "❌ REJECTED"
                                print(f"❌ AI rejected signal for {sym}")
                            
                            # Extract AI confidence if present
                            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', ai_response)
                            if conf_match:
                                ai_conf = int(conf_match.group(1))
                                ai_boost = max(ai_boost, (ai_conf - 50) // 5)

                        final_confidence = max(0, min(95, int(local_signal.get("confidence", 0) + ai_boost)))

                        if final_confidence >= SIGNAL_CONF_THRESHOLD:
                            signals_found += 1
                            
                            patterns_text = ", ".join(local_signal.get('patterns', [])[:3]) or "None"
                            
                            msg = (
                                f"<b>🎯 {sym} {local_signal['side']} SIGNAL</b>\n\n"
                                f"<b>📊 Trade Setup:</b>\n"
                                f"💰 Entry: <code>{fmt_price(local_signal['entry'])}</code>\n"
                                f"🛑 Stop Loss: <code>{fmt_price(local_signal['sl'])}</code>\n"
                                f"🎯 Take Profit: <code>{fmt_price(local_signal['tp'])}</code>\n\n"
                                f"<b>📈 Analysis:</b>\n"
                                f"✨ Confidence: <b>{final_confidence}%</b>\n"
                                f"⚖️ Risk/Reward: <b>1:{local_signal.get('rr', 0)}</b>\n"
                                f"📊 Score: <b>{local_signal.get('score', 0)}</b>\n\n"
                                f"<b>🔍 Detected Patterns:</b>\n"
                                f"{patterns_text}\n\n"
                                f"<b>💡 Reason:</b>\n"
                                f"{local_signal.get('reason', '')[:140]}\n\n"
                                f"<b>🤖 AI Verdict:</b> {ai_verdict}"
                            )

                            chart_path = plot_signal_chart(sym, candles_raw, local_signal)
                            await send_photo(session, msg, chart_path)
                            print(f"⚡ SIGNAL SENT: {sym} {local_signal['side']} | Conf: {final_confidence}%")
                        else:
                            print(f"📉 {sym}: Below threshold after AI ({final_confidence}%)")
                    else:
                        reason = local_signal.get('reason', 'no setup')
                        conf = local_signal.get('confidence', 0)
                        patterns = ", ".join(local_signal.get('patterns', [])[:2]) or "none"
                        print(f"📊 {sym}: No signal (Conf: {conf}%, Patterns: {patterns}, Reason: {reason[:50]})")

                except Exception as e:
                    print(f"❌ Error analyzing {sym}: {e}")
                    traceback.print_exc()

            # Summary message
            print(f"\n{'='*70}")
            print(f"📊 Iteration {iteration} complete: {signals_found} signals found")
            print(f"{'='*70}")
            
            if signals_found == 0:
                # Create detailed summary
                top_by_conf = sorted(analysis_summary, key=lambda x: x['confidence'], reverse=True)[:5]
                
                summary_lines = [f"<b>📊 Scan #{iteration} Complete</b>\n"]
                summary_lines.append(f"✅ {len(SYMBOLS)} symbols analyzed")
                summary_lines.append(f"🎯 {signals_found} signals above {int(SIGNAL_CONF_THRESHOLD)}% threshold")
                summary_lines.append(f"⏰ Next scan in {POLL_INTERVAL//60} minutes\n")
                summary_lines.append(f"<b>🔝 Top 5 by Confidence:</b>")
                
                for item in top_by_conf:
                    patterns_txt = ", ".join(item['patterns'][:2]) if item['patterns'] else "—"
                    summary_lines.append(
                        f"• {item['symbol']}: {item['confidence']}% | {item['side'].upper()} | {patterns_txt[:30]}"
                    )
                
                await send_text(session, "\n".join(summary_lines))

            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
