#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v5.2 (Optimized with better rate limiting)
import os, asyncio, aiohttp, traceback, numpy as np, json, logging, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple
import sqlite3
from contextlib import asynccontextmanager

load_dotenv()

# ------------- LOGGING -------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------- CONFIG -------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AAVEUSDT",
    "TRXUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT"
]

# Optimized settings
POLL_INTERVAL = max(10, int(os.getenv("POLL_INTERVAL", 900)))  # Reduced default to 15 min
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))  # Slightly lower threshold
MAX_SIGNALS_PER_HOUR = int(os.getenv("MAX_SIGNALS_PER_HOUR", 8))  # Increased limit
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_URL = "https://api.binance.com/api/v3"
CANDLE_URL = f"{BASE_URL}/klines?symbol={{symbol}}&interval={{interval}}&limit={{limit}}"

# ------------- DATABASE -------------
class SignalDatabase:
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        try:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON signals(symbol, timestamp);")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database init error: {e}")
    
    def save_signal(self, signal_data: Dict):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO signals (symbol, side, entry_price, sl_price, tp_price, confidence, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data['symbol'],
                signal_data['side'],
                float(signal_data['entry']),
                float(signal_data['sl']),
                float(signal_data['tp']),
                float(signal_data['confidence']),
                signal_data['reason']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB save failed: {e}")
    
    def get_recent_signals_count(self, hours=1) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = datetime.now() - timedelta(hours=hours)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE timestamp > ? AND status = 'active'",
                (cutoff,)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"DB count failed: {e}")
            return 0

db = SignalDatabase()

# ------------- OPTIMIZED RATE LIMITER -------------
class OptimizedRateLimiter:
    """
    Smart rate limiter that adapts to Binance limits:
    - 1200 requests per minute (20 per second)
    - Weight-based limiting
    - Batch processing support
    """
    def __init__(self, requests_per_minute=1200, burst_size=10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = []
        self._lock = asyncio.Lock()
        self.last_log_time = 0.0
        self.log_suppress_interval = 10.0  # Log every 10 seconds max
    
    async def wait_if_needed(self, weight=1):
        """Wait if needed based on request weight"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests (older than 1 minute)
            minute_ago = now - 60
            self.requests = [req_time for req_time in self.requests if req_time > minute_ago]
            
            # Check if we can make the request
            if len(self.requests) < self.requests_per_minute:
                self.requests.append(now)
                return
            
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = 60 - (now - oldest_request)
            
            if wait_time > 0:
                if now - self.last_log_time > self.log_suppress_interval:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                    self.last_log_time = now
                
                await asyncio.sleep(wait_time)
                
                # Update after waiting
                now2 = time.time()
                minute_ago2 = now2 - 60
                self.requests = [req_time for req_time in self.requests if req_time > minute_ago2]
                self.requests.append(now2)

# Use optimized rate limiter - much more permissive
rate_limiter = OptimizedRateLimiter(requests_per_minute=600, burst_size=20)

# Global backoff for 429 errors - reduced default
remote_backoff_until = 0.0
REMOTE_BACKOFF_DEFAULT = 10  # Reduced from 30 to 10 seconds

# ------------- UTILITIES / INDICATORS -------------
def fmt_price(p):
    try:
        p = float(p)
        if abs(p) < 0.001:
            return f"{p:.8f}"
        elif abs(p) < 1:
            return f"{p:.6f}"
        elif abs(p) < 100:
            return f"{p:.4f}"
        else:
            return f"{p:.2f}"
    except Exception:
        return str(p)

def calculate_rsi(closes, period=RSI_PERIOD):
    """Fast RSI calculation"""
    if not closes or len(closes) < period + 1:
        return []
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use numpy for faster calculation
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi_values = []
    
    for i in range(period, len(closes)):
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
        
        # Update averages with new data point
        if i < len(deltas):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
    
    return rsi_values

def ema(values, period):
    """Vectorized EMA calculation"""
    if not values or len(values) < period:
        return []
    
    values_array = np.array(values, dtype=float)
    
    # Initialize
    k = 2.0 / (period + 1)
    ema_values = [None] * (period - 1)
    
    # Start with SMA
    sma = np.mean(values_array[:period])
    ema_values.append(sma)
    
    # Calculate EMA
    current_ema = sma
    for i in range(period, len(values)):
        current_ema = values_array[i] * k + current_ema * (1 - k)
        ema_values.append(current_ema)
    
    return ema_values

def calculate_macd(closes):
    """Optimized MACD calculation"""
    if not closes or len(closes) < MACD_SLOW:
        return {"macd": [], "signal": [], "histogram": []}
    
    try:
        fast_ema = ema(closes, MACD_FAST)
        slow_ema = ema(closes, MACD_SLOW)
        
        # Calculate MACD line
        macd_line = []
        min_length = min(len(fast_ema), len(slow_ema))
        
        for i in range(min_length):
            if fast_ema[i] is not None and slow_ema[i] is not None:
                macd_line.append(fast_ema[i] - slow_ema[i])
        
        if not macd_line:
            return {"macd": [], "signal": [], "histogram": []}
        
        # Signal line
        signal_line = ema(macd_line, MACD_SIGNAL) if len(macd_line) >= MACD_SIGNAL else []
        signal_clean = [x for x in signal_line if x is not None] if signal_line else []
        
        # Histogram
        histogram = []
        for i in range(min(len(macd_line), len(signal_clean))):
            histogram.append(macd_line[i] - signal_clean[i])
        
        return {
            "macd": macd_line,
            "signal": signal_clean,
            "histogram": histogram
        }
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {"macd": [], "signal": [], "histogram": []}

def horizontal_levels(closes, highs, lows, lookback=30, binsize=0.003):
    """Optimized support/resistance detection"""
    if not closes:
        return []
    
    lookback = min(lookback, len(closes))
    
    try:
        # Combine all price points
        recent_closes = closes[-lookback:]
        recent_highs = highs[-lookback:] if highs else []
        recent_lows = lows[-lookback:] if lows else []
        
        all_points = recent_closes + recent_highs + recent_lows
        all_points = [p for p in all_points if p is not None and p > 0]
        
        if not all_points:
            return []
        
        # Use numpy for faster processing
        points_array = np.array(all_points)
        levels = []
        
        for point in points_array:
            found = False
            for level in levels:
                if abs((level["price"] - point) / point) < binsize:
                    level["count"] += 1
                    level["price"] = (level["price"] * (level["count"] - 1) + point) / level["count"]
                    found = True
                    break
            
            if not found:
                levels.append({"price": float(point), "count": 1})
        
        # Filter and sort
        significant_levels = [lv["price"] for lv in levels if lv["count"] >= 2]
        return sorted(significant_levels)[:5]
        
    except Exception as e:
        logger.error(f"Level detection error: {e}")
        return []

# ------------- OPTIMIZED ANALYSIS -------------
def analyze_trade_logic(candles, rr_min=1.3):  # Reduced R/R requirement
    """Optimized trade analysis with better error handling"""
    try:
        if not candles or len(candles) < 30:  # Reduced requirement
            return {"side": "none", "confidence": 0, "reason": "insufficient data"}
        
        # Clean and convert candles
        cleaned = []
        for c in candles:
            try:
                if not c or len(c) < 5:
                    continue
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                if all(x > 0 for x in [o, h, l, cl]):  # Valid prices
                    cleaned.append([o, h, l, cl])
            except Exception:
                continue
        
        if len(cleaned) < 30:
            return {"side": "none", "confidence": 0, "reason": "insufficient valid candles"}
        
        # Extract price arrays
        closes = [x[3] for x in cleaned]
        highs = [x[1] for x in cleaned]
        lows = [x[2] for x in cleaned]
        current_price = closes[-1]
        
        # Calculate indicators
        ema_9 = ema(closes, 9)
        ema_21 = ema(closes, 21)
        ema_50 = ema(closes, 50) if len(closes) >= 50 else ema_21
        rsi_values = calculate_rsi(closes)
        macd = calculate_macd(closes)
        
        # Validate indicators
        if not ema_9 or not ema_21 or not rsi_values:
            return {"side": "none", "confidence": 0, "reason": "indicator calculation failed"}
        
        current_ema_9 = ema_9[-1] if ema_9[-1] is not None else current_price
        current_ema_21 = ema_21[-1] if ema_21[-1] is not None else current_price
        current_ema_50 = (ema_50[-1] if ema_50 and ema_50[-1] is not None else current_ema_21)
        current_rsi = rsi_values[-1] if rsi_values else 50
        
        # Support and resistance
        levels = horizontal_levels(closes, highs, lows, lookback=20)  # Reduced lookback
        support = max([lv for lv in levels if lv < current_price], default=None) if levels else None
        resistance = min([lv for lv in levels if lv > current_price], default=None) if levels else None
        
        # Confidence scoring
        confidence = 50
        reasons = []
        
        # EMA trend analysis (simplified)
        if current_price > current_ema_9 > current_ema_21:
            confidence += 15
            reasons.append("bullish EMA alignment")
            trend_direction = "up"
        elif current_price < current_ema_9 < current_ema_21:
            confidence += 15
            reasons.append("bearish EMA alignment")
            trend_direction = "down"
        else:
            confidence -= 5
            reasons.append("mixed EMA signals")
            trend_direction = "neutral"
        
        # RSI analysis
        if 30 <= current_rsi <= 70:  # Not extreme
            confidence += 8
            reasons.append(f"healthy RSI({current_rsi:.1f})")
        elif current_rsi > 70:
            confidence += 5
            reasons.append(f"overbought RSI({current_rsi:.1f})")
        elif current_rsi < 30:
            confidence += 5
            reasons.append(f"oversold RSI({current_rsi:.1f})")
        
        # MACD confirmation
        if macd.get("macd") and macd.get("signal") and len(macd["macd"]) >= 2:
            macd_current = macd["macd"][-1]
            signal_current = macd["signal"][-1] if macd["signal"] else 0
            
            if macd_current > signal_current:
                confidence += 8
                reasons.append("MACD above signal")
            else:
                confidence += 3
                reasons.append("MACD below signal")
        
        # Generate signals with relaxed conditions
        if trend_direction == "up" and confidence >= 65:
            entry_price = current_price
            
            # Dynamic stop loss
            if support:
                stop_loss = support * 0.997
            else:
                stop_loss = current_price * 0.97  # 3% stop
            
            # Dynamic take profit
            if resistance:
                take_profit = resistance * 0.998
            else:
                take_profit = current_price * 1.06  # 6% target
            
            # Check R/R ratio
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio >= rr_min:
                    return {
                        "side": "BUY",
                        "entry": entry_price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "confidence": min(95, confidence + 10),
                        "reason": "; ".join(reasons),
                        "risk_reward": rr_ratio,
                        "indicators": {
                            "rsi": current_rsi,
                            "ema_9": current_ema_9,
                            "ema_21": current_ema_21
                        }
                    }
        
        elif trend_direction == "down" and confidence >= 65:
            entry_price = current_price
            
            # Dynamic stop loss
            if resistance:
                stop_loss = resistance * 1.003
            else:
                stop_loss = current_price * 1.03  # 3% stop
            
            # Dynamic take profit
            if support:
                take_profit = support * 1.002
            else:
                take_profit = current_price * 0.94  # 6% target
            
            # Check R/R ratio
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio >= rr_min:
                    return {
                        "side": "SELL",
                        "entry": entry_price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "confidence": min(95, confidence + 10),
                        "reason": "; ".join(reasons),
                        "risk_reward": rr_ratio,
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
                "rsi": current_rsi,
                "ema_9": current_ema_9,
                "ema_21": current_ema_21
            }
        }
        
    except Exception as e:
        logger.error(f"Trade analysis error: {e}")
        return {"side": "none", "confidence": 0, "reason": f"analysis error: {str(e)[:50]}"}

def multi_tf_confirmation(c30m, c1h, c4h=None):
    """Multi-timeframe with relaxed requirements"""
    try:
        signal_30m = analyze_trade_logic(c30m)
        signal_1h = analyze_trade_logic(c1h)
        
        # More lenient confirmation
        if signal_30m.get("side") == "none":
            return signal_30m
        
        if signal_1h.get("side") == "none":
            # Allow 30m signal if confidence is high
            if signal_30m.get("confidence", 0) >= 80:
                signal_30m["reason"] += "; single TF (high conf)"
                return signal_30m
            return {"side": "none", "confidence": 0, "reason": "no 1h confirmation"}
        
        if signal_30m["side"] == signal_1h["side"]:
            # Aligned signals - boost confidence
            boost = 15
            if c4h:
                signal_4h = analyze_trade_logic(c4h)
                if signal_4h.get("side") == signal_30m["side"]:
                    boost += 10
            
            enhanced_signal = signal_30m.copy()
            enhanced_signal["confidence"] = min(100, signal_30m.get("confidence", 0) + boost)
            enhanced_signal["reason"] += f"; TF aligned (1h conf: {signal_1h.get('confidence', 0)}%)"
            
            return enhanced_signal
        
        # Conflicting signals - check if 30m is very strong
        if signal_30m.get("confidence", 0) >= 85:
            signal_30m["reason"] += f"; TF conflict but high 30m conf"
            return signal_30m
        
        return {
            "side": "none",
            "confidence": 0,
            "reason": f"TF conflict: 30m={signal_30m.get('side')}, 1h={signal_1h.get('side')}"
        }
        
    except Exception as e:
        logger.error(f"Multi-TF analysis error: {e}")
        return {"side": "none", "confidence": 0, "reason": "multi-tf error"}

# ------------- ENHANCED CHART -------------
def plot_enhanced_chart(symbol, candles, signal):
    """Create better charts with more info"""
    try:
        # Process last 40 candles for better visibility
        recent_candles = candles[-40:]
        dates = []
        opens, highs, lows, closes = [], [], [], []
        
        for candle in recent_candles:
            try:
                timestamp = int(candle[0])
                dates.append(datetime.utcfromtimestamp(timestamp / 1000))
                opens.append(float(candle[1]))
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
            except Exception:
                continue
        
        if not closes:
            return plot_fallback_chart(symbol, signal)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main price chart
        x_indices = range(len(dates))
        
        # Plot candlesticks
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = 'green' if c >= o else 'red'
            # Wick
            ax1.plot([i, i], [l, h], color='black', linewidth=0.8, alpha=0.8)
            # Body
            body_height = abs(c - o)
            body_bottom = min(o, c)
            ax1.bar(i, body_height, bottom=body_bottom, width=0.6, 
                   color=color, alpha=0.7)
        
        # Add EMAs
        if len(closes) >= 21:
            ema_9_vals = ema(closes, 9)
            ema_21_vals = ema(closes, 21)
            
            if ema_9_vals:
                valid_ema9 = [(i, val) for i, val in enumerate(ema_9_vals) 
                             if val is not None and i < len(x_indices)]
                if valid_ema9:
                    ema9_x, ema9_y = zip(*valid_ema9)
                    ax1.plot(ema9_x, ema9_y, color='orange', 
                            label='EMA 9', linewidth=2, alpha=0.8)
            
            if ema_21_vals:
                valid_ema21 = [(i, val) for i, val in enumerate(ema_21_vals) 
                              if val is not None and i < len(x_indices)]
                if valid_ema21:
                    ema21_x, ema21_y = zip(*valid_ema21)
                    ax1.plot(ema21_x, ema21_y, color='blue', 
                            label='EMA 21', linewidth=2, alpha=0.8)
        
        # Signal levels
        if signal.get("side") != "none":
            ax1.axhline(signal["entry"], color="purple", linestyle="--", 
                       alpha=0.9, linewidth=2, label=f"Entry: {fmt_price(signal['entry'])}")
            ax1.axhline(signal["sl"], color="red", linestyle="--", 
                       alpha=0.9, linewidth=2, label=f"SL: {fmt_price(signal['sl'])}")
            ax1.axhline(signal["tp"], color="green", linestyle="--", 
                       alpha=0.9, linewidth=2, label=f"TP: {fmt_price(signal['tp'])}")
        
        # Chart formatting
        ax1.set_title(f"{symbol} - {signal.get('side', 'NONE')} | "
                     f"Confidence: {signal.get('confidence', 0):.1f}% | "
                     f"R/R: {signal.get('risk_reward', 0):.2f}", 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price', fontsize=12)
        
        # Format x-axis
        if len(dates) > 0:
            step = max(1, len(dates) // 8)
            tick_positions = list(range(0, len(dates), step))
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels([dates[i].strftime('%m/%d %H:%M') 
                               for i in tick_positions], rotation=45)
        
        # RSI subplot
        if len(closes) >= RSI_PERIOD:
            rsi_values = calculate_rsi(closes)
            if rsi_values:
                rsi_x = range(len(closes) - len(rsi_values), len(closes))
                ax2.plot(rsi_x, rsi_values, color='purple', linewidth=2, label='RSI')
                ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
                ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
                ax2.axhline(50, color='gray', linestyle='-', alpha=0.3)
                ax2.set_ylabel('RSI', fontsize=10)
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save chart
        temp_file = NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Enhanced chart creation error: {e}")
        return plot_fallback_chart(symbol, signal)

def plot_fallback_chart(symbol, signal):
    """Simple fallback chart"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"{symbol}\n{signal.get('side', 'NO SIGNAL')}\n"
                          f"Conf: {signal.get('confidence', 0):.1f}%", 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=16, fontweight='bold')
        ax.set_title(f"{symbol} Trading Signal", fontsize=14)
        
        temp_file = NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return temp_file.name
    except Exception:
        return None

# ------------- OPTIMIZED HTTP SESSION -------------
@asynccontextmanager
async def get_session():
    """Optimized session with better connection pooling"""
    connector = aiohttp.TCPConnector(
        limit=50,  # Increased connection pool
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=25, connect=8)
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=
