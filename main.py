#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v5.0 (Major Improvements)
import os, re, asyncio, aiohttp, traceback, numpy as np, json, logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple
import sqlite3
from contextlib import asynccontextmanager
import time

load_dotenv()

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
MAX_SIGNALS_PER_HOUR = int(os.getenv("MAX_SIGNALS_PER_HOUR", 5))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# API URLs with better versioning
BASE_URL = "https://api.binance.com/api/v3"
TICKER_URL = f"{BASE_URL}/ticker/24hr?symbol={{symbol}}"
CANDLE_URL = f"{BASE_URL}/klines?symbol={{symbol}}&interval={{interval}}&limit={{limit}}"

# ---------------- DATABASE SETUP ----------------
class SignalDatabase:
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                tp_price REAL NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON signals(symbol, timestamp);
        """)
        conn.commit()
        conn.close()
    
    def save_signal(self, signal_data: Dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signals (symbol, side, entry_price, sl_price, tp_price, confidence, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data['symbol'],
            signal_data['side'],
            signal_data['entry'],
            signal_data['sl'],
            signal_data['tp'],
            signal_data['confidence'],
            signal_data['reason']
        ))
        conn.commit()
        conn.close()
    
    def get_recent_signals_count(self, hours=1) -> int:
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now() - timedelta(hours=hours)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM signals 
            WHERE timestamp > ? AND status = 'active'
        """, (cutoff,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

db = SignalDatabase()

# ---------------- RATE LIMITING ----------------
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    async def wait_if_needed(self):
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.window_seconds - (now - self.requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

rate_limiter = RateLimiter()

# ---------------- ENHANCED UTILS ----------------
def fmt_price(p: float) -> str:
    if abs(p) < 0.001:
        return f"{p:.8f}"
    elif abs(p) < 1:
        return f"{p:.6f}"
    elif abs(p) < 100:
        return f"{p:.4f}"
    else:
        return f"{p:.2f}"

def calculate_rsi(closes: List[float], period: int = RSI_PERIOD) -> List[float]:
    """Calculate RSI indicator"""
    if len(closes) < period + 1:
        return []
    
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    for i in range(period, len(closes)):
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
        
        # Update averages
        if i < len(deltas):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
    
    return rsi_values

def calculate_macd(closes: List[float]) -> Dict[str, List[float]]:
    """Calculate MACD indicator"""
    if len(closes) < MACD_SLOW:
        return {"macd": [], "signal": [], "histogram": []}
    
    ema_fast = ema(closes, MACD_FAST)
    ema_slow = ema(closes, MACD_SLOW)
    
    macd_line = []
    start_idx = max(len(ema_fast) - len([x for x in ema_fast if x is not None]),
                   len(ema_slow) - len([x for x in ema_slow if x is not None]))
    
    for i in range(start_idx, len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
    
    signal_line = ema(macd_line, MACD_SIGNAL)
    
    histogram = []
    signal_start = len(signal_line) - len([x for x in signal_line if x is not None])
    for i in range(signal_start, len(macd_line)):
        if i < len(signal_line) and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
    
    return {
        "macd": macd_line,
        "signal": [x for x in signal_line if x is not None],
        "histogram": histogram
    }

def ema(values: List[float], period: int) -> List[float]:
    """Enhanced EMA calculation with better handling"""
    if len(values) < period:
        return []
    
    k = 2 / (period + 1)
    ema_values = [None] * (period - 1)
    
    # Initialize with SMA
    sma = sum(values[:period]) / period
    ema_values.append(sma)
    
    # Calculate EMA
    for i in range(period, len(values)):
        ema_val = values[i] * k + ema_values[-1] * (1 - k)
        ema_values.append(ema_val)
    
    return ema_values

def detect_pattern(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Detect common chart patterns"""
    if len(closes) < 20:
        return {"pattern": "none", "confidence": 0}
    
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    recent_closes = closes[-10:]
    
    # Double top pattern
    if len(recent_highs) >= 5:
        max_high = max(recent_highs)
        high_indices = [i for i, h in enumerate(recent_highs) if abs(h - max_high) / max_high < 0.02]
        if len(high_indices) >= 2 and high_indices[-1] - high_indices[0] >= 3:
            return {"pattern": "double_top", "confidence": 70}
    
    # Double bottom pattern
    if len(recent_lows) >= 5:
        min_low = min(recent_lows)
        low_indices = [i for i, l in enumerate(recent_lows) if abs(l - min_low) / min_low < 0.02]
        if len(low_indices) >= 2 and low_indices[-1] - low_indices[0] >= 3:
            return {"pattern": "double_bottom", "confidence": 70}
    
    return {"pattern": "none", "confidence": 0}

def horizontal_levels(closes: List[float], highs: List[float], lows: List[float], 
                     lookback: int = 50, binsize: float = 0.002) -> List[float]:
    """Enhanced support/resistance level detection"""
    if len(closes) < lookback:
        lookback = len(closes)
    
    points = closes[-lookback:] + highs[-lookback:] + lows[-lookback:]
    levels = []
    
    for point in points:
        found = False
        for level in levels:
            if abs((level["price"] - point) / point) < binsize:
                level["count"] += 1
                level["price"] = (level["price"] * (level["count"] - 1) + point) / level["count"]
                found = True
                break
        if not found:
            levels.append({"price": point, "count": 1})
    
    # Sort by count and filter significant levels
    levels.sort(key=lambda x: -x["count"])
    significant_levels = [lv["price"] for lv in levels if lv["count"] >= 3][:5]
    
    return significant_levels

def analyze_trade_logic(candles: List[List[float]], rr_min: float = 1.5) -> Dict:
    """Enhanced trade analysis with multiple indicators"""
    if len(candles) < 50:
        return {"side": "none", "confidence": 0, "reason": "insufficient data"}
    
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    
    current_price = closes[-1]
    
    # Calculate indicators
    ema_9 = ema(closes, 9)
    ema_21 = ema(closes, 21)
    ema_50 = ema(closes, 50)
    rsi_values = calculate_rsi(closes)
    macd_data = calculate_macd(closes)
    
    if not ema_9 or not ema_21 or not rsi_values:
        return {"side": "none", "confidence": 0, "reason": "indicator calculation failed"}
    
    current_ema_9 = ema_9[-1]
    current_ema_21 = ema_21[-1]
    current_ema_50 = ema_50[-1] if ema_50 else current_ema_21
    current_rsi = rsi_values[-1] if rsi_values else 50
    
    # Support and resistance levels
    levels = horizontal_levels(closes, highs, lows)
    support = max([lv for lv in levels if lv < current_price], default=None)
    resistance = min([lv for lv in levels if lv > current_price], default=None)
    
    # Pattern detection
    pattern_info = detect_pattern(highs, lows, closes)
    
    confidence = 50
    reasons = []
    
    # EMA trend analysis
    if current_price > current_ema_9 > current_ema_21 > current_ema_50:
        confidence += 15
        reasons.append("strong uptrend (EMAs aligned)")
    elif current_price < current_ema_9 < current_ema_21 < current_ema_50:
        confidence += 15
        reasons.append("strong downtrend (EMAs aligned)")
    elif current_price > current_ema_9 > current_ema_21:
        confidence += 10
        reasons.append("uptrend (short-term EMAs aligned)")
    elif current_price < current_ema_9 < current_ema_21:
        confidence += 10
        reasons.append("downtrend (short-term EMAs aligned)")
    else:
        confidence -= 5
        reasons.append("mixed EMA signals")
    
    # RSI analysis
    if current_rsi > 70:
        confidence += 5
        reasons.append(f"overbought RSI({current_rsi:.1f})")
    elif current_rsi < 30:
        confidence += 5
        reasons.append(f"oversold RSI({current_rsi:.1f})")
    
    # MACD analysis
    if macd_data["macd"] and macd_data["signal"]:
        macd_current = macd_data["macd"][-1]
        signal_current = macd_data["signal"][-1]
        
        if macd_current > signal_current and len(macd_data["macd"]) > 1:
            prev_macd = macd_data["macd"][-2]
            prev_signal = macd_data["signal"][-1] if len(macd_data["signal"]) > 1 else signal_current
            
            if prev_macd <= prev_signal:  # Bullish crossover
                confidence += 10
                reasons.append("MACD bullish crossover")
        elif macd_current < signal_current and len(macd_data["macd"]) > 1:
            prev_macd = macd_data["macd"][-2]
            prev_signal = macd_data["signal"][-1] if len(macd_data["signal"]) > 1 else signal_current
            
            if prev_macd >= prev_signal:  # Bearish crossover
                confidence += 10
                reasons.append("MACD bearish crossover")
    
    # Pattern bonus
    if pattern_info["pattern"] != "none":
        confidence += pattern_info["confidence"] * 0.2
        reasons.append(f"pattern: {pattern_info['pattern']}")
    
    # Generate signals
    if (current_price > current_ema_9 and current_rsi < 70 and 
        support and resistance and confidence >= 60):
        
        entry_price = current_price
        stop_loss = support * 0.995 if support else current_price * 0.97
        take_profit = resistance * 0.995 if resistance else current_price * 1.06
        
        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
        
        if risk_reward >= rr_min:
            return {
                "side": "BUY",
                "entry": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "confidence": min(95, confidence + 10),
                "reason": "; ".join(reasons),
                "risk_reward": risk_reward,
                "indicators": {
                    "rsi": current_rsi,
                    "ema_9": current_ema_9,
                    "ema_21": current_ema_21
                }
            }
    
    elif (current_price < current_ema_9 and current_rsi > 30 and 
          support and resistance and confidence >= 60):
        
        entry_price = current_price
        stop_loss = resistance * 1.005 if resistance else current_price * 1.03
        take_profit = support * 1.005 if support else current_price * 0.94
        
        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
        
        if risk_reward >= rr_min:
            return {
                "side": "SELL",
                "entry": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "confidence": min(95, confidence + 10),
                "reason": "; ".join(reasons),
                "risk_reward": risk_reward,
                "indicators": {
                    "rsi": current_rsi,
                    "ema_9": current_ema_9,
                    "ema_21": current_ema_21
                }
            }
    
    return {
        "side": "none",
        "confidence": confidence,
        "reason": "; ".join(reasons),
        "indicators": {
            "rsi": current_rsi if 'current_rsi' in locals() else 50,
            "ema_9": current_ema_9 if 'current_ema_9' in locals() else current_price,
            "ema_21": current_ema_21 if 'current_ema_21' in locals() else current_price
        }
    }

def multi_tf_confirmation(c30m: List[List[float]], c1h: List[List[float]], c4h: List[List[float]] = None) -> Dict:
    """Enhanced multi-timeframe analysis"""
    signal_30m = analyze_trade_logic(c30m)
    signal_1h = analyze_trade_logic(c1h)
    
    if signal_30m["side"] == "none" or signal_1h["side"] == "none":
        return {"side": "none", "confidence": 0, "reason": "no clear signals on timeframes"}
    
    if signal_30m["side"] == signal_1h["side"]:
        # Boost confidence for aligned signals
        confidence_boost = 20
        if c4h:
            signal_4h = analyze_trade_logic(c4h)
            if signal_4h["side"] == signal_30m["side"]:
                confidence_boost += 10
        
        enhanced_signal = signal_30m.copy()
        enhanced_signal["confidence"] = min(100, signal_30m["confidence"] + confidence_boost)
        enhanced_signal["reason"] += f"; aligned with 1h TF (conf: {signal_1h['confidence']}%)"
        
        return enhanced_signal
    
    return {
        "side": "none", 
        "confidence": 0, 
        "reason": f"TF conflict: 30m={signal_30m['side']}, 1h={signal_1h['side']}"
    }

# ---------------- ENHANCED CHART ----------------
def plot_advanced_chart(symbol: str, candles: List[List], signal: Dict, indicators: Dict = None) -> str:
    """Create advanced chart with indicators"""
    try:
        dates = [datetime.utcfromtimestamp(int(x[0])/1000) for x in candles[-50:]]  # Last 50 candles
        opens = [float(x[1]) for x in candles[-50:]]
        highs = [float(x[2]) for x in candles[-50:]]
        lows = [float(x[3]) for x in candles[-50:]]
        closes = [float(x[4]) for x in candles[-50:]]
        volumes = [float(x[5]) for x in candles[-50:]]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                           gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Candlestick chart
        for i, (d, o, h, l, c) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = 'green' if c >= o else 'red'
            ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax1.plot([i-0.3, i+0.3], [o, o], color=color, linewidth=2)
            ax1.plot([i-0.3, i+0.3], [c, c], color=color, linewidth=2)
        
        # EMAs
        if len(closes) >= 21:
            ema_9_vals = ema(closes, 9)
            ema_21_vals = ema(closes, 21)
            
            if ema_9_vals:
                valid_ema9 = [(i, val) for i, val in enumerate(ema_9_vals) if val is not None]
                if valid_ema9:
                    ema9_x, ema9_y = zip(*valid_ema9)
                    ax1.plot(ema9_x, ema9_y, color='orange', label='EMA 9', linewidth=1.5)
            
            if ema_21_vals:
                valid_ema21 = [(i, val) for i, val in enumerate(ema_21_vals) if val is not None]
                if valid_ema21:
                    ema21_x, ema21_y = zip(*valid_ema21)
                    ax1.plot(ema21_x, ema21_y, color='blue', label='EMA 21', linewidth=1.5)
        
        # Signal levels
        if signal["side"] != "none":
            ax1.axhline(signal["entry"], color="blue", linestyle="--", alpha=0.8, 
                       label=f"Entry: {fmt_price(signal['entry'])}")
            ax1.axhline(signal["sl"], color="red", linestyle="--", alpha=0.8,
                       label=f"SL: {fmt_price(signal['sl'])}")
            ax1.axhline(signal["tp"], color="green", linestyle="--", alpha=0.8,
                       label=f"TP: {fmt_price(signal['tp'])}")
        
        ax1.set_title(f"{symbol} - {signal['side']} Signal | Confidence: {signal['confidence']:.1f}% | "
                     f"R/R: {signal.get('risk_reward', 0):.2f}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, len(dates), max(1, len(dates)//10)))
        ax1.set_xticklabels([dates[i].strftime('%m/%d %H:%M') for i in range(0, len(dates), max(1, len(dates)//10))], 
                           rotation=45)
        
        # RSI subplot
        if len(closes) >= RSI_PERIOD:
            rsi_values = calculate_rsi(closes)
            if rsi_values:
                rsi_x = range(len(closes) - len(rsi_values), len(closes))
                ax2.plot(rsi_x, rsi_values, color='purple', linewidth=2)
                ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
                ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
        
        # Volume subplot
        ax3.bar(range(len(volumes)), volumes, color=['green' if closes[i] >= opens[i] else 'red' for i in range(len(closes))],
               alpha=0.6)
        ax3.set_ylabel('Volume')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return tmp.name
        
    except Exception as e:
        logger.error(f"Chart creation error for {symbol}: {e}")
        # Fallback to simple chart
        return plot_simple_chart(symbol, candles, signal)

def plot_simple_chart(symbol: str, candles: List[List], signal: Dict) -> str:
    """Fallback simple chart"""
    dates = [datetime.utcfromtimestamp(int(x[0])/1000) for x in candles[-30:]]
    closes = [float(x[4]) for x in candles[-30:]]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, closes, linewidth=2, color='blue')
    
    if signal["side"] != "none":
        plt.axhline(signal["entry"], color="blue", linestyle="--", label=f"Entry: {fmt_price(signal['entry'])}")
        plt.axhline(signal["sl"], color="red", linestyle="--", label=f"SL: {fmt_price(signal['sl'])}")
        plt.axhline(signal["tp"], color="green", linestyle="--", label=f"TP: {fmt_price(signal['tp'])}")
    
    plt.title(f"{symbol} - {signal['side']} | Confidence: {signal['confidence']:.1f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight')
    plt.close()
    
    return tmp.name

# ---------------- ENHANCED FETCH ----------------
@asynccontextmanager
async def get_session():
    """Context manager for aiohttp session with retry logic"""
    connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300, use_dns_cache=True)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        yield session

async def fetch_json_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[Dict]:
    """Fetch JSON with retry logic and rate limiting"""
    await rate_limiter.wait_if_needed()
    
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    
    return None

# ---------------- ENHANCED AI ----------------
async def get_enhanced_ai_analysis(symbol: str, candles: List[List], signal: Dict) -> Optional[str]:
    """Enhanced AI analysis with better prompting"""
    if not client:
        return None
    
    try:
        # Prepare market data summary
        closes = [float(c[4]) for c in candles[-20:]]
        current_price = closes[-1]
        price_change_24h = ((current_price - closes[0]) / closes[0]) * 100
        
        market_summary = {
            "symbol": symbol,
            "current_price": current_price,
            "price_change_24h": f"{price_change_24h:.2f}%",
            "local_signal": {
                "side": signal["side"],
                "confidence": signal["confidence"],
                "entry": signal.get("entry"),
                "risk_reward": signal.get("risk_reward")
            },
            "indicators": signal.get("indicators", {})
        }
        
        system_prompt = """You are an expert crypto trader and technical analyst. 
        Analyze the provided market data and determine if there's a high-confidence trading opportunity.
        
        Focus on:
        - Technical indicator alignment
        - Risk/reward ratio (minimum 1.5:1)
        - Market structure and price action
        - Support/resistance levels
        
        Only suggest signals with 75%+ confidence.
        Format: SYMBOL - BUY/SELL - ENTRY:price - SL:price - TP:price - CONF:xx% - REASON:brief_explanation"""
        
        user_prompt = f"""Market Data Analysis:
        {json.dumps(market_summary, indent=2)}
        
        Current local analysis suggests: {signal['side']} with {signal['confidence']}% confidence.
        Please provide your independent analysis and confirm/reject this signal."""
        
        loop = asyncio.get_running_loop()
        
        def call_openai():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
        
        response = await loop.run_in_executor(None, call_openai)
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return None

# ---------------- ENHANCED TELEGRAM ----------------
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, session: aiohttp.ClientSession, text: str, parse_mode: str = "Markdown"):
        """Send text message with formatting"""
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram config, logging: {text}")
            return
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:4096],  # Telegram message limit
            "parse_mode": parse_mode
        }
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Telegram send message failed: {response.status}")
        except Exception as e:
            logger.error(f"Telegram message error: {e}")
    
    async def send_photo(self, session: aiohttp.ClientSession, caption: str, photo_path: str):
        """Send photo with caption"""
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram config, logging: {caption}")
            return
        
        url = f"{self.base_url}/sendPhoto"
        
        try:
            with open(photo_path, 'rb') as photo_file:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('caption', caption[:1024])  # Caption limit
                data.add_field('photo', photo_file, filename='chart.png')
                
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        logger.error(f"Telegram send photo failed: {response.status}")
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(photo_path)
            except:
                pass

telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# ---------------- MARKET SCANNER ----------------
class MarketScanner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.last_scan_time = None
        self.scan_results = {}
    
    async def scan_all_symbols(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Scan all symbols for trading opportunities"""
        self.last_scan_time = datetime.now()
        signals = []
        
        # Check signal rate limiting
        recent_signals = db.get_recent_signals_count(hours=1)
        if recent_signals >= MAX_SIGNALS_PER_HOUR:
            logger.warning(f"Signal rate limit reached: {recent_signals}/{MAX_SIGNALS_PER_HOUR} per hour")
            return []
        
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.analyze_symbol(session, symbol))
            tasks.append(task)
        
        # Process symbols concurrently with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        async def limited_analyze(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_analyze(task) for task in tasks], 
                                     return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {self.symbols[i]}: {result}")
                continue
            
            if result and result.get("side") != "none":
                result["symbol"] = self.symbols[i]
                signals.append(result)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return signals[:3]  # Return top 3 signals
    
    async def analyze_symbol(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """Analyze individual symbol"""
        try:
            # Fetch multiple timeframes
            candles_30m = await fetch_json_with_retry(
                session, CANDLE_URL.format(symbol=symbol, interval="30m", limit=100)
            )
            candles_1h = await fetch_json_with_retry(
                session, CANDLE_URL.format(symbol=symbol, interval="1h", limit=100)
            )
            candles_4h = await fetch_json_with_retry(
                session, CANDLE_URL.format(symbol=symbol, interval="4h", limit=50)
            )
            
            if not candles_30m or not candles_1h:
                logger.warning(f"Failed to fetch candles for {symbol}")
                return None
            
            # Convert to proper format
            c30m = [[float(x) for x in candle[:6]] for candle in candles_30m]
            c1h = [[float(x) for x in candle[:6]] for candle in candles_1h]
            c4h = [[float(x) for x in candle[:6]] for candle in candles_4h] if candles_4h else None
            
            # Multi-timeframe analysis
            signal = multi_tf_confirmation(c30m, c1h, c4h)
            
            if signal["side"] != "none" and signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                # Get AI confirmation
                ai_analysis = await get_enhanced_ai_analysis(symbol, c30m, signal)
                
                # Boost confidence if AI confirms
                if ai_analysis and "NO_SIGNAL" not in ai_analysis.upper():
                    signal["confidence"] = min(100, signal["confidence"] + 5)
                    signal["ai_analysis"] = ai_analysis
                
                # Create chart
                chart_path = plot_advanced_chart(symbol, c30m, signal)
                signal["chart_path"] = chart_path
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Symbol analysis error for {symbol}: {e}")
            return None

# ---------------- PERFORMANCE MONITOR ----------------
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.scan_count = 0
        self.signal_count = 0
        self.error_count = 0
    
    def log_scan(self):
        self.scan_count += 1
    
    def log_signal(self):
        self.signal_count += 1
    
    def log_error(self):
        self.error_count += 1
    
    def get_stats(self) -> str:
        uptime = time.time() - self.start_time
        return (f"📊 *Bot Statistics*\n"
                f"⏱ Uptime: {uptime/3600:.1f} hours\n"
                f"🔍 Scans completed: {self.scan_count}\n"
                f"⚡ Signals generated: {self.signal_count}\n"
                f"❌ Errors: {self.error_count}\n"
                f"📈 Success rate: {((self.scan_count-self.error_count)/max(1,self.scan_count)*100):.1f}%")

monitor = PerformanceMonitor()

# ---------------- MAIN ENHANCED LOOP ----------------
async def enhanced_trading_loop():
    """Main enhanced trading loop with better error handling"""
    scanner = MarketScanner(SYMBOLS)
    
    async with get_session() as session:
        # Startup notification
        startup_msg = (f"🚀 *Enhanced Trading Bot v5.0 Started!*\n\n"
                      f"📊 Monitoring: {len(SYMBOLS)} symbols\n"
                      f"⏱ Scan interval: {POLL_INTERVAL}s\n"
                      f"🎯 Confidence threshold: {SIGNAL_CONF_THRESHOLD}%\n"
                      f"⚡ Max signals/hour: {MAX_SIGNALS_PER_HOUR}\n"
                      f"🤖 AI analysis: {'✅' if client else '❌'}\n\n"
                      f"Ready to hunt for profitable trades! 🎯")
        
        logger.info("Trading bot started successfully!")
        await telegram.send_message(session, startup_msg)
        
        iteration = 0
        last_stats_time = time.time()
        
        while True:
            iteration += 1
            scan_start_time = time.time()
            
            try:
                logger.info(f"\n=== SCAN {iteration} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                monitor.log_scan()
                
                # Scan for signals
                signals = await scanner.scan_all_symbols(session)
                
                scan_duration = time.time() - scan_start_time
                logger.info(f"Scan completed in {scan_duration:.2f}s, found {len(signals)} signals")
                
                # Process signals
                for signal in signals:
                    try:
                        await process_signal(session, signal)
                        monitor.log_signal()
                        
                        # Rate limiting between signals
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error processing signal for {signal.get('symbol', 'unknown')}: {e}")
                        monitor.log_error()
                
                # Send periodic stats (every 4 hours)
                if time.time() - last_stats_time > 14400:  # 4 hours
                    stats_msg = monitor.get_stats()
                    await telegram.send_message(session, stats_msg)
                    last_stats_time = time.time()
                
                # Dynamic sleep based on signal activity
                sleep_time = POLL_INTERVAL
                if len(signals) > 0:
                    sleep_time = max(300, POLL_INTERVAL // 2)  # Faster scans if signals found
                
                logger.info(f"Sleeping for {sleep_time}s until next scan...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                error_msg = f"❌ *Scan Error #{iteration}*\n`{str(e)[:500]}`"
                logger.error(f"Critical scan error: {e}")
                logger.error(traceback.format_exc())
                
                await telegram.send_message(session, error_msg)
                monitor.log_error()
                
                # Exponential backoff on errors
                await asyncio.sleep(min(300, 30 * 2**(monitor.error_count % 4)))

async def process_signal(session: aiohttp.ClientSession, signal: Dict):
    """Process and send individual signal"""
    symbol = signal["symbol"]
    
    # Format message
    rr_ratio = signal.get("risk_reward", 0)
    indicators = signal.get("indicators", {})
    
    message = (f"🎯 *{symbol} {signal['side']} SIGNAL*\n\n"
              f"💰 Entry: `{fmt_price(signal['entry'])}`\n"
              f"🛑 Stop Loss: `{fmt_price(signal['sl'])}`\n"
              f"🎯 Take Profit: `{fmt_price(signal['tp'])}`\n"
              f"📊 Confidence: *{signal['confidence']:.1f}%*\n"
              f"⚖️ Risk/Reward: *{rr_ratio:.2f}*\n\n"
              f"📈 *Indicators:*\n"
              f"• RSI: {indicators.get('rsi', 0):.1f}\n"
              f"• EMA9: {fmt_price(indicators.get('ema_9', 0))}\n"
              f"• EMA21: {fmt_price(indicators.get('ema_21', 0))}\n\n"
              f"📝 *Reason:* {signal['reason'][:200]}...\n\n"
              f"⏰ {datetime.now().strftime('%H:%M:%S UTC')}")
    
    # Add AI analysis if available
    if signal.get("ai_analysis"):
        message += f"\n\n🤖 *AI Analysis:*\n`{signal['ai_analysis'][:150]}...`"
    
    # Send chart with signal
    chart_path = signal.get("chart_path")
    if chart_path:
        await telegram.send_photo(session, message, chart_path)
    else:
        await telegram.send_message(session, message)
    
    # Save to database
    db.save_signal({
        "symbol": symbol,
        "side": signal["side"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "confidence": signal["confidence"],
        "reason": signal["reason"]
    })
    
    logger.info(f"✅ Signal sent: {symbol} {signal['side']} @ {fmt_price(signal['entry'])} "
               f"(Conf: {signal['confidence']:.1f}%, R/R: {rr_ratio:.2f})")

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    try:
        logger.info("Starting Enhanced Crypto Trading Bot v5.0")
        asyncio.run(enhanced_trading_loop())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
