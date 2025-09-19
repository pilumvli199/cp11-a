#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v5.0 (Patched: defensive fixes for None, rate-limit log suppression)
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
import math

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

# API URLs
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

# ---------------- RATE LIMITER (with log suppression) ----------------
class RateLimiter:
    """
    Concurrency-safe sliding-window rate limiter with log suppression
    to avoid flooding logs when many coroutines wait.
    """
    def __init__(self, max_requests=8, window_seconds=60, log_suppress_interval=1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
        self.last_log_time = 0.0
        self.log_suppress_interval = log_suppress_interval

    async def wait_if_needed(self):
        async with self._lock:
            now = time.time()
            # remove expired
            self.requests = [t for t in self.requests if now - t < self.window_seconds]
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return
            oldest = self.requests[0]
            sleep_time = self.window_seconds - (now - oldest)
            if sleep_time <= 0:
                self.requests = [t for t in self.requests if now - t < self.window_seconds]
                self.requests.append(now)
                return
            # log but suppress too-frequent logs
            if now - self.last_log_time > self.log_suppress_interval:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f}s")
                self.last_log_time = now
        await asyncio.sleep(sleep_time)
        async with self._lock:
            now2 = time.time()
            self.requests = [t for t in self.requests if now2 - t < self.window_seconds]
            self.requests.append(now2)

rate_limiter = RateLimiter(max_requests=8, window_seconds=60, log_suppress_interval=1.0)

# ---------------- UTILITIES ----------------
def fmt_price(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return str(p)
    if abs(p) < 0.001:
        return f"{p:.8f}"
    elif abs(p) < 1:
        return f"{p:.6f}"
    elif abs(p) < 100:
        return f"{p:.4f}"
    else:
        return f"{p:.2f}"

def calculate_rsi(closes: List[float], period: int = RSI_PERIOD) -> List[float]:
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
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
        if i < len(deltas):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
    return rsi_values

def ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema_values = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema_values.append(sma)
    for i in range(period, len(values)):
        ema_val = values[i] * k + ema_values[-1] * (1 - k)
        ema_values.append(ema_val)
    return ema_values

def calculate_macd(closes: List[float]) -> Dict[str, List[float]]:
    if len(closes) < MACD_SLOW:
        return {"macd": [], "signal": [], "histogram": []}
    ema_fast = ema(closes, MACD_FAST)
    ema_slow = ema(closes, MACD_SLOW)
    macd_line = []
    for i in range(len(closes)):
        f = ema_fast[i] if i < len(ema_fast) else None
        s = ema_slow[i] if i < len(ema_slow) else None
        if f is not None and s is not None:
            macd_line.append(f - s)
    signal_line = ema(macd_line, MACD_SIGNAL) if macd_line else []
    signal_clean = [x for x in signal_line if x is not None]
    histogram = []
    for idx in range(min(len(macd_line), len(signal_clean))):
        histogram.append(macd_line[idx] - signal_clean[idx])
    return {"macd": macd_line, "signal": signal_clean, "histogram": histogram}

def detect_pattern(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    if len(closes) < 20:
        return {"pattern": "none", "confidence": 0}
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    if len(recent_highs) >= 5:
        max_high = max(recent_highs)
        high_indices = [i for i, h in enumerate(recent_highs) if abs(h - max_high) / max_high < 0.02]
        if len(high_indices) >= 2 and high_indices[-1] - high_indices[0] >= 3:
            return {"pattern": "double_top", "confidence": 70}
    if len(recent_lows) >= 5:
        min_low = min(recent_lows)
        low_indices = [i for i, l in enumerate(recent_lows) if abs(l - min_low) / min_low < 0.02]
        if len(low_indices) >= 2 and low_indices[-1] - low_indices[0] >= 3:
            return {"pattern": "double_bottom", "confidence": 70}
    return {"pattern": "none", "confidence": 0}

def horizontal_levels(closes: List[float], highs: List[float], lows: List[float], lookback: int = 50, binsize: float = 0.002) -> List[float]:
    if len(closes) < lookback:
        lookback = len(closes)
    points = []
    try:
        points = closes[-lookback:] + highs[-lookback:] + lows[-lookback:]
    except Exception:
        points = [p for p in (closes[-lookback:] + highs[-lookback:] + lows[-lookback:]) if p is not None]
    levels = []
    for point in points:
        if point is None or point == 0:
            continue
        found = False
        for level in levels:
            try:
                if abs((level["price"] - point) / point) < binsize:
                    level["count"] += 1
                    level["price"] = (level["price"] * (level["count"] - 1) + point) / level["count"]
                    found = True
                    break
            except Exception:
                continue
        if not found:
            levels.append({"price": point, "count": 1})
    levels.sort(key=lambda x: -x["count"])
    significant_levels = [lv["price"] for lv in levels if lv["count"] >= 3][:5]
    return significant_levels

# ---------------- SAFE analyze_trade_logic (defensive) ----------------
def analyze_trade_logic(candles: List[List[float]], rr_min: float = 1.5) -> Dict:
    """Enhanced trade analysis with safe guards against None values"""
    try:
        if not candles or len(candles) < 50:
            return {"side": "none", "confidence": 0, "reason": "insufficient data"}
        # ensure each candle has at least 5 elements
        filtered = []
        for c in candles:
            if not c or len(c) < 5:
                continue
            try:
                # cast close/high/low to float safely
                o = float(c[1]); h = float(c[2]); l = float(c[3]); cl = float(c[4])
                filtered.append([o, h, l, cl])
            except Exception:
                continue
        if len(filtered) < 50:
            return {"side": "none", "confidence": 0, "reason": "insufficient valid candle rows after cleaning"}
        closes = [c[3] for c in filtered]
        highs = [c[1] for c in filtered]
        lows = [c[2] for c in filtered]
        current_price = closes[-1]
        # indicators
        ema_9 = ema(closes, 9)
        ema_21 = ema(closes, 21)
        ema_50 = ema(closes, 50)
        rsi_values = calculate_rsi(closes)
        macd_data = calculate_macd(closes)
        if not ema_9 or not ema_21 or not rsi_values:
            return {"side": "none", "confidence": 0, "reason": "indicator calculation failed"}
        # last EMA values may still be None in edge cases — guard
        current_ema_9 = ema_9[-1] if ema_9 and ema_9[-1] is not None else None
        current_ema_21 = ema_21[-1] if ema_21 and ema_21[-1] is not None else None
        current_ema_50 = ema_50[-1] if ema_50 and ema_50[-1] is not None else current_ema_21
        current_rsi = rsi_values[-1] if rsi_values else None
        if current_ema_9 is None or current_ema_21 is None or current_rsi is None:
            return {"side": "none", "confidence": 0, "reason": "insufficient indicator values (None found)"}
        levels = horizontal_levels(closes, highs, lows)
        support = max([lv for lv in levels if lv < current_price], default=None) if levels else None
        resistance = min([lv for lv in levels if lv > current_price], default=None) if levels else None
        pattern_info = detect_pattern(highs, lows, closes)
        confidence = 50
        reasons = []
        # EMA trend analysis
        try:
            if current_price > current_ema_9 > current_ema_21 > current_ema_50:
                confidence += 15; reasons.append("strong uptrend (EMAs aligned)")
            elif current_price < current_ema_9 < current_ema_21 < current_ema_50:
                confidence += 15; reasons.append("strong downtrend (EMAs aligned)")
            elif current_price > current_ema_9 > current_ema_21:
                confidence += 10; reasons.append("uptrend (short-term EMAs aligned)")
            elif current_price < current_ema_9 < current_ema_21:
                confidence += 10; reasons.append("downtrend (short-term EMAs aligned)")
            else:
                confidence -= 5; reasons.append("mixed EMA signals")
        except Exception:
            reasons.append("ema ordering check failed")
        # RSI
        if current_rsi is not None:
            if current_rsi > 70:
                confidence += 5; reasons.append(f"overbought RSI({current_rsi:.1f})")
            elif current_rsi < 30:
                confidence += 5; reasons.append(f"oversold RSI({current_rsi:.1f})")
        # MACD
        try:
            if macd_data["macd"] and macd_data["signal"]:
                macd_current = macd_data["macd"][-1]
                signal_current = macd_data["signal"][-1]
                if macd_current > signal_current and len(macd_data["macd"]) > 1:
                    prev_macd = macd_data["macd"][-2]
                    prev_signal = macd_data["signal"][-1] if len(macd_data["signal"]) > 1 else signal_current
                    if prev_macd <= prev_signal:
                        confidence += 10; reasons.append("MACD bullish crossover")
                elif macd_current < signal_current and len(macd_data["macd"]) > 1:
                    prev_macd = macd_data["macd"][-2]
                    prev_signal = macd_data["signal"][-1] if len(macd_data["signal"]) > 1 else signal_current
                    if prev_macd >= prev_signal:
                        confidence += 10; reasons.append("MACD bearish crossover")
        except Exception:
            reasons.append("macd check failed")
        # pattern
        if pattern_info["pattern"] != "none":
            confidence += pattern_info["confidence"] * 0.2
            reasons.append(f"pattern: {pattern_info['pattern']}")
        # Build BUY signal safely
        if (current_price is not None and current_ema_9 is not None and current_rsi is not None and
            support is not None and resistance is not None and confidence >= 60):
            try:
                entry_price = float(current_price)
                stop_loss = float(support * 0.995) if support else None
                take_profit = float(resistance * 0.995) if resistance else None
                if stop_loss is None or take_profit is None:
                    return {"side": "none", "confidence": confidence, "reason": "support/resistance missing for trade calc"}
                denom = (entry_price - stop_loss)
                if denom == 0:
                    return {"side": "none", "confidence": confidence, "reason": "zero risk (entry equals stop)"}
                risk_reward = (take_profit - entry_price) / denom
                if risk_reward >= rr_min:
                    return {
                        "side": "BUY", "entry": entry_price, "sl": stop_loss, "tp": take_profit,
                        "confidence": min(95, confidence + 10), "reason": "; ".join(reasons),
                        "risk_reward": risk_reward, "indicators": {"rsi": current_rsi, "ema_9": current_ema_9, "ema_21": current_ema_21}
                    }
            except Exception as e:
                reasons.append(f"buy calc error: {e}")
        # Build SELL signal safely
        if (current_price is not None and current_ema_9 is not None and current_rsi is not None and
            support is not None and resistance is not None and confidence >= 60):
            try:
                entry_price = float(current_price)
                stop_loss = float(resistance * 1.005) if resistance else None
                take_profit = float(support * 1.005) if support else None
                if stop_loss is None or take_profit is None:
                    return {"side": "none", "confidence": confidence, "reason": "support/resistance missing for sell calc"}
                denom = (stop_loss - entry_price)
                if denom == 0:
                    return {"side": "none", "confidence": confidence, "reason": "zero risk (stop equals entry)"}
                risk_reward = (entry_price - take_profit) / denom
                if risk_reward >= rr_min:
                    return {
                        "side": "SELL", "entry": entry_price, "sl": stop_loss, "tp": take_profit,
                        "confidence": min(95, confidence + 10), "reason": "; ".join(reasons),
                        "risk_reward": risk_reward, "indicators": {"rsi": current_rsi, "ema_9": current_ema_9, "ema_21": current_ema_21}
                    }
            except Exception as e:
                reasons.append(f"sell calc error: {e}")
        return {"side": "none", "confidence": confidence, "reason": "; ".join(reasons),
                "indicators": {"rsi": current_rsi or 50, "ema_9": current_ema_9 or current_price, "ema_21": current_ema_21 or current_price}}
    except Exception as ex:
        logger.error(f"analyze_trade_logic fatal: {ex}")
        logger.error(traceback.format_exc())
        return {"side": "none", "confidence": 0, "reason": f"internal error: {ex}"}

def multi_tf_confirmation(c30m: List[List[float]], c1h: List[List[float]], c4h: List[List[float]] = None) -> Dict:
    signal_30m = analyze_trade_logic(c30m)
    signal_1h = analyze_trade_logic(c1h)
    if signal_30m.get("side") == "none" or signal_1h.get("side") == "none":
        return {"side": "none", "confidence": 0, "reason": "no clear signals on timeframes"}
    if signal_30m["side"] == signal_1h["side"]:
        confidence_boost = 20
        if c4h:
            signal_4h = analyze_trade_logic(c4h)
            if signal_4h.get("side") == signal_30m["side"]:
                confidence_boost += 10
        enhanced_signal = signal_30m.copy()
        enhanced_signal["confidence"] = min(100, signal_30m.get("confidence", 0) + confidence_boost)
        enhanced_signal["reason"] = enhanced_signal.get("reason", "") + f"; aligned with 1h TF (conf: {signal_1h.get('confidence',0)}%)"
        return enhanced_signal
    return {"side": "none", "confidence": 0, "reason": f"TF conflict: 30m={signal_30m.get('side')},1h={signal_1h.get('side')}"}

# ---------------- CHARTS, FETCH, AI, TELEGRAM, SCANNER, MONITOR -- unchanged except small safety tweaks ----------------
# For brevity: reuse your previous robust implementations but ensure defensive guards exist when using values.
# Below I include the major ones unchanged except small guards (you can keep the rest of your existing code).

def plot_advanced_chart(symbol: str, candles: List[List], signal: Dict, indicators: Dict = None) -> str:
    try:
        # Use last up to 50 candles that have valid numeric values
        good = []
        for row in candles[-100:]:
            try:
                t = int(row[0]); o=float(row[1]); h=float(row[2]); l=float(row[3]); c=float(row[4]); v=float(row[5])
                good.append([t,o,h,l,c,v])
            except Exception:
                continue
        if len(good) < 8:
            return plot_simple_chart(symbol, candles, signal)
        dates = [datetime.utcfromtimestamp(int(x[0])/1000) for x in good[-50:]]
        opens = [x[1] for x in good[-50:]]
        highs = [x[2] for x in good[-50:]]
        lows = [x[3] for x in good[-50:]]
        closes = [x[4] for x in good[-50:]]
        volumes = [x[5] for x in good[-50:]]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        for i, (o,h,l,c) in enumerate(zip(opens, highs, lows, closes)):
            color = 'green' if c >= o else 'red'
            ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax1.plot([i-0.3, i+0.3], [o, o], color=color, linewidth=2)
            ax1.plot([i-0.3, i+0.3], [c, c], color=color, linewidth=2)
        if len(closes) >= 21:
            ema9 = ema(closes, 9); ema21 = ema(closes, 21)
            if ema9:
                valid = [(i,val) for i,val in enumerate(ema9) if val is not None]
                if valid:
                    x_vals, y_vals = zip(*valid); ax1.plot(x_vals, y_vals, linewidth=1.5, label='EMA9')
            if ema21:
                valid = [(i,val) for i,val in enumerate(ema21) if val is not None]
                if valid:
                    x_vals, y_vals = zip(*valid); ax1.plot(x_vals, y_vals, linewidth=1.5, label='EMA21')
        if signal.get("side") != "none":
            try:
                ax1.axhline(signal["entry"], color="blue", linestyle="--", alpha=0.8, label=f"Entry: {fmt_price(signal['entry'])}")
                ax1.axhline(signal["sl"], color="red", linestyle="--", alpha=0.8, label=f"SL: {fmt_price(signal['sl'])}")
                ax1.axhline(signal["tp"], color="green", linestyle="--", alpha=0.8, label=f"TP: {fmt_price(signal['tp'])}")
            except Exception:
                pass
        ax1.set_title(f"{symbol} - {signal.get('side','none')} | Conf: {signal.get('confidence',0):.1f}%")
        ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
        if len(closes) >= RSI_PERIOD:
            rsi_vals = calculate_rsi(closes)
            if rsi_vals:
                rsi_x = range(len(closes)-len(rsi_vals), len(closes)); ax2.plot(rsi_x, rsi_vals, linewidth=1.5); ax2.axhline(70, linestyle='--'); ax2.axhline(30, linestyle='--'); ax2.set_ylim(0,100)
        ax3.bar(range(len(volumes)), volumes, alpha=0.6)
        plt.tight_layout()
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches='tight', dpi=200)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        logger.error(f"plot_advanced_chart error: {e}")
        return plot_simple_chart(symbol, candles, signal)

def plot_simple_chart(symbol: str, candles: List[List], signal: Dict) -> str:
    try:
        good = []
        for row in candles[-60:]:
            try:
                t=int(row[0]); c=float(row[4]); good.append((t,c))
            except Exception:
                continue
        if not good:
            tmp = NamedTemporaryFile(delete=False, suffix=".png"); plt.figure(); plt.text(0.5,0.5,"No data",ha='center'); plt.savefig(tmp.name); plt.close(); return tmp.name
        dates = [datetime.utcfromtimestamp(int(t)/1000) for t,c in good]
        closes = [c for t,c in good]
        plt.figure(figsize=(10,5)); plt.plot(dates, closes); plt.title(f"{symbol} - {signal.get('side','none')}"); plt.grid(True)
        if signal.get("side") != "none":
            try:
                plt.axhline(signal["entry"], linestyle='--'); plt.axhline(signal["sl"], linestyle='--'); plt.axhline(signal["tp"], linestyle='--')
            except Exception:
                pass
        tmp = NamedTemporaryFile(delete=False, suffix=".png"); plt.savefig(tmp.name, bbox_inches='tight'); plt.close(); return tmp.name
    except Exception as e:
        logger.error(f"plot_simple_chart failed: {e}")
        tmp = NamedTemporaryFile(delete=False, suffix=".png"); plt.figure(); plt.text(0.5,0.5,"Chart error",ha='center'); plt.savefig(tmp.name); plt.close(); return tmp.name

@asynccontextmanager
async def get_session():
    connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300, use_dns_cache=True)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        yield session

async def fetch_json_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[Dict]:
    await rate_limiter.wait_if_needed()
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited by remote, waiting {wait_time}s (attempt {attempt+1})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url} (attempt {attempt+1})")
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None

async def get_enhanced_ai_analysis(symbol: str, candles: List[List], signal: Dict) -> Optional[str]:
    if not client:
        return None
    try:
        closes = []
        for c in candles[-20:]:
            try:
                closes.append(float(c[4]))
            except:
                continue
        if not closes:
            return None
        current_price = closes[-1]
        price_change_24h = ((current_price - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
        market_summary = {
            "symbol": symbol, "current_price": current_price,
            "price_change_24h": f"{price_change_24h:.2f}%", "local_signal": {"side": signal.get("side"), "confidence": signal.get("confidence"), "entry": signal.get("entry"), "risk_reward": signal.get("risk_reward")}, "indicators": signal.get("indicators", {})
        }
        system_prompt = "You are an expert crypto trader..."
        user_prompt = f"Market Data Analysis:\n{json.dumps(market_summary, indent=2)}\nPlease provide your independent analysis."
        loop = asyncio.get_running_loop()
        def call_openai():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                max_tokens=300, temperature=0.2
            )
        resp = await loop.run_in_executor(None, call_openai)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return None

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token; self.chat_id = chat_id; self.base_url = f"https://api.telegram.org/bot{bot_token}"
    async def send_message(self, session: aiohttp.ClientSession, text: str, parse_mode: str = "Markdown"):
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram config, log message: {text[:120]}")
            return
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text[:4096], "parse_mode": parse_mode}
        try:
            async with session.post(url, json=payload) as r:
                if r.status != 200:
                    logger.error(f"Telegram send message failed: {r.status}")
        except Exception as e:
            logger.error(f"Telegram message error: {e}")
    async def send_photo(self, session: aiohttp.ClientSession, caption: str, photo_path: str):
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram config, log caption: {caption[:120]}")
            try: os.unlink(photo_path)
            except: pass
            return
        url = f"{self.base_url}/sendPhoto"
        try:
            with open(photo_path, 'rb') as photo_file:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('caption', caption[:1024])
                data.add_field('photo', photo_file, filename='chart.png')
                async with session.post(url, data=data) as r:
                    if r.status != 200:
                        logger.error(f"Telegram send photo failed: {r.status}")
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
        finally:
            try: os.unlink(photo_path)
            except: pass

telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

class MarketScanner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols; self.last_scan_time = None; self.scan_results = {}
    async def scan_all_symbols(self, session: aiohttp.ClientSession) -> List[Dict]:
        self.last_scan_time = datetime.now(); signals=[]
        recent_signals = db.get_recent_signals_count(hours=1)
        if recent_signals >= MAX_SIGNALS_PER_HOUR:
            logger.warning(f"Signal rate limit reached: {recent_signals}/{MAX_SIGNALS_PER_HOUR} per hour")
            return []
        tasks=[]
        for symbol in self.symbols:
            tasks.append(asyncio.create_task(self.analyze_symbol(session, symbol)))
            await asyncio.sleep(0.12)
        sem = asyncio.Semaphore(3)
        async def limited(t): 
            async with sem:
                return await t
        results = await asyncio.gather(*[limited(t) for t in tasks], return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Error analyzing {self.symbols[i]}: {res}")
                continue
            if res and res.get("side") != "none":
                res["symbol"] = self.symbols[i]; signals.append(res)
        signals.sort(key=lambda x: x.get("confidence",0), reverse=True)
        return signals[:3]
    async def analyze_symbol(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        try:
            c30 = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="30m", limit=100))
            c1 = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="1h", limit=100))
            c4 = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="4h", limit=50))
            if not c30 or not c1:
                logger.warning(f"Failed to fetch candles for {symbol}")
                return None
            c30m = [[float(x) for x in candle[:6]] for candle in c30 if candle and len(candle) >= 5]
            c1h = [[float(x) for x in candle[:6]] for candle in c1 if candle and len(candle) >= 5]
            c4h = [[float(x) for x in candle[:6]] for candle in c4 if c4 and len(c4) and len(candle) >= 5] if c4 else None
            signal = multi_tf_confirmation(c30m, c1h, c4h)
            if signal.get("side") != "none" and signal.get("confidence",0) >= SIGNAL_CONF_THRESHOLD:
                ai_analysis = await get_enhanced_ai_analysis(symbol, c30m, signal)
                if ai_analysis and "NO_SIGNAL" not in ai_analysis.upper():
                    signal["confidence"] = min(100, signal.get("confidence",0) + 5)
                    signal["ai_analysis"] = ai_analysis
                chart_path = plot_advanced_chart(symbol, c30m, signal)
                signal["chart_path"] = chart_path
                return signal
            return None
        except Exception as e:
            logger.error(f"Symbol analysis error for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time(); self.scan_count=0; self.signal_count=0; self.error_count=0
    def log_scan(self): self.scan_count += 1
    def log_signal(self): self.signal_count += 1
    def log_error(self): self.error_count += 1
    def get_stats(self) -> str:
        uptime = time.time() - self.start_time
        return (f"📊 *Bot Statistics*\n⏱ Uptime: {uptime/3600:.1f} hours\n🔍 Scans: {self.scan_count}\n⚡ Signals: {self.signal_count}\n❌ Errors: {self.error_count}\n")

monitor = PerformanceMonitor()

async def process_signal(session: aiohttp.ClientSession, signal: Dict):
    symbol = signal.get("symbol","UNKNOWN"); rr_ratio = signal.get("risk_reward",0); indicators = signal.get("indicators",{})
    message = (f"🎯 *{symbol} {signal.get('side','')} SIGNAL*\n\n"
               f"💰 Entry: `{fmt_price(signal.get('entry','N/A'))}`\n"
               f"🛑 Stop Loss: `{fmt_price(signal.get('sl','N/A'))}`\n"
               f"🎯 Take Profit: `{fmt_price(signal.get('tp','N/A'))}`\n"
               f"📊 Confidence: *{signal.get('confidence',0):.1f}%*\n"
               f"⚖️ R/R: *{rr_ratio:.2f}*\n\n"
               f"📈 RSI: {indicators.get('rsi',0):.1f} • EMA9: {fmt_price(indicators.get('ema_9',0))} • EMA21: {fmt_price(indicators.get('ema_21',0))}\n\n"
               f"📝 {signal.get('reason','')[:400]}\n⏰ {datetime.now().strftime('%H:%M:%S UTC')}")
    try:
        chart_path = signal.get("chart_path")
        if chart_path and os.path.exists(chart_path):
            await telegram.send_photo(session, message, chart_path)
        else:
            await telegram.send_message(session, message)
        if all(k in signal for k in ("symbol","side","entry","sl","tp","confidence","reason")):
            db.save_signal({"symbol": signal["symbol"], "side": signal["side"], "entry": signal["entry"], "sl": signal["sl"], "tp": signal["tp"], "confidence": signal["confidence"], "reason": signal["reason"]})
        logger.info(f"✅ Signal sent: {symbol} {signal.get('side')} @ {fmt_price(signal.get('entry'))} (Conf:{signal.get('confidence')}, R/R:{rr_ratio:.2f})")
    except Exception as e:
        logger.error(f"Failed to process/send signal for {symbol}: {e}")
        logger.error(traceback.format_exc())

async def enhanced_trading_loop():
    scanner = MarketScanner(SYMBOLS)
    async with get_session() as session:
        startup_msg = (f"🚀 *Enhanced Trading Bot v5.0 Started!*\n\n📊 Monitoring: {len(SYMBOLS)} symbols\n⏱ Scan interval: {POLL_INTERVAL}s\n🎯 Confidence threshold: {SIGNAL_CONF_THRESHOLD}%\n⚡ Max signals/hour: {MAX_SIGNALS_PER_HOUR}\n🤖 AI analysis: {'✅' if client else '❌'}")
        logger.info("Trading bot started successfully!")
        await telegram.send_message(session, startup_msg)
        iteration=0; last_stats=time.time()
        while True:
            iteration+=1; scan_start=time.time()
            try:
                logger.info(f"\n=== SCAN {iteration} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                monitor.log_scan()
                signals = await scanner.scan_all_symbols(session)
                logger.info(f"Scan completed in {(time.time()-scan_start):.2f}s, found {len(signals)} signals")
                for sig in signals:
                    try:
                        await process_signal(session, sig); monitor.log_signal(); await asyncio.sleep(2)
                    except Exception as e:
                        logger.error(f"Error processing signal: {e}"); monitor.log_error()
                if time.time() - last_stats > 14400:
                    await telegram.send_message(session, monitor.get_stats()); last_stats = time.time()
                sleep_time = POLL_INTERVAL if not signals else max(300, POLL_INTERVAL//2)
                logger.info(f"Sleeping for {sleep_time}s until next scan...")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Critical scan error: {e}"); logger.error(traceback.format_exc())
                await telegram.send_message(session, f"❌ *Scan Error #{iteration}* ` {str(e)[:400]} `")
                monitor.log_error()
                await asyncio.sleep(min(300, 30 * 2**(monitor.error_count % 4)))

if __name__ == "__main__":
    try:
        logger.info("Starting Enhanced Crypto Trading Bot v5.0")
        asyncio.run(enhanced_trading_loop())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
