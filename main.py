#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v4.1 (Improved & More Reliable)
import os, asyncio, aiohttp, traceback, json, logging, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional
import sqlite3

load_dotenv()

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AAVEUSDT",
    "TRXUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT"
]
POLL_INTERVAL = max(60, int(os.getenv("POLL_INTERVAL", 1800)))  # Minimum 1 minute
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 65.0))  # Lowered threshold
MAX_SIGNALS_PER_HOUR = int(os.getenv("MAX_SIGNALS_PER_HOUR", 5))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_URL = "https://api.binance.com/api/v3"
CANDLE_URL = f"{BASE_URL}/klines?symbol={{symbol}}&interval={{interval}}&limit={{limit}}"

# ---------------- DATABASE ----------------
class SignalDatabase:
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, side TEXT NOT NULL,
            entry_price REAL NOT NULL, sl_price REAL NOT NULL, tp_price REAL NOT NULL,
            confidence REAL NOT NULL, reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()
    
    def save_signal(self, signal_data: Dict):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""INSERT INTO signals (symbol, side, entry_price, sl_price, tp_price, confidence, reason)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                signal_data['symbol'], signal_data['side'], float(signal_data['entry']),
                float(signal_data['sl']), float(signal_data['tp']), 
                float(signal_data['confidence']), signal_data['reason']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB save failed: {e}")
    
    def get_recent_signals_count(self, hours=1) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = datetime.now() - timedelta(hours=hours)
            cur = conn.execute("SELECT COUNT(*) FROM signals WHERE timestamp > ?", (cutoff,))
            count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"DB count failed: {e}")
            return 0

db = SignalDatabase()

# ---------------- RATE LIMITER ----------------
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self._lock:
            now = time.time()
            # Remove old requests
            self.requests = [t for t in self.requests if now - t < self.window_seconds]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return
            
            # Need to wait
            oldest = self.requests[0]
            sleep_time = self.window_seconds - (now - oldest)
            if sleep_time > 0:
                logger.info(f"Rate limit hit, sleeping {sleep_time:.1f}s")
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            async with self._lock:
                self.requests.append(time.time())

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# ---------------- UTILITIES ----------------
def fmt_price(p):
    try:
        p = float(p)
        if abs(p) < 0.001: return f"{p:.8f}"
        elif abs(p) < 1: return f"{p:.6f}"
        elif abs(p) < 100: return f"{p:.4f}"
        else: return f"{p:.2f}"
    except:
        return str(p)

def calculate_rsi(closes, period=RSI_PERIOD):
    if len(closes) < period + 1:
        return []
    
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_vals = []
    for i in range(period, len(closes)):
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_vals.append(rsi)
        
        if i < len(deltas):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
    
    return rsi_vals

def ema(values, period):
    if len(values) < period:
        return []
    
    k = 2 / (period + 1)
    ema_vals = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema_vals.append(sma)
    
    for i in range(period, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    
    return ema_vals

def find_support_resistance(closes, highs, lows, lookback=50):
    """Find support and resistance levels using pivot points"""
    if len(closes) < lookback:
        lookback = len(closes)
    
    recent_closes = closes[-lookback:]
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    current_price = closes[-1]
    
    # Find potential support (recent lows below current price)
    supports = [low for low in recent_lows if low < current_price * 0.98]
    support = max(supports) if supports else current_price * 0.95
    
    # Find potential resistance (recent highs above current price)
    resistances = [high for high in recent_highs if high > current_price * 1.02]
    resistance = min(resistances) if resistances else current_price * 1.05
    
    return support, resistance

def analyze_trade_logic(candles, rr_min=1.2):
    """Enhanced trade analysis with multiple indicators"""
    try:
        if not candles or len(candles) < 30:
            return {"side": "none", "confidence": 0, "reason": "insufficient data"}
        
        # Extract OHLC data
        opens = [float(c[1]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]
        
        current_price = closes[-1]
        
        # Calculate indicators
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        ema50 = ema(closes, 50)
        rsi_vals = calculate_rsi(closes)
        
        if not ema9 or not ema21 or not rsi_vals:
            return {"side": "none", "confidence": 0, "reason": "indicator calculation failed"}
        
        cur_ema9 = ema9[-1]
        cur_ema21 = ema21[-1]
        cur_ema50 = ema50[-1] if ema50 else cur_ema21
        cur_rsi = rsi_vals[-1]
        
        # Find support and resistance
        support, resistance = find_support_resistance(closes, highs, lows)
        
        confidence = 40
        reasons = []
        
        # Trend analysis
        if current_price > cur_ema9 > cur_ema21:
            confidence += 20
            reasons.append("strong bullish trend")
        elif current_price < cur_ema9 < cur_ema21:
            confidence += 20
            reasons.append("strong bearish trend")
        elif current_price > cur_ema9:
            confidence += 10
            reasons.append("short-term bullish")
        elif current_price < cur_ema9:
            confidence += 10
            reasons.append("short-term bearish")
        
        # RSI analysis
        if cur_rsi < 35:
            confidence += 15
            reasons.append(f"oversold RSI({cur_rsi:.1f})")
        elif cur_rsi > 65:
            confidence += 15
            reasons.append(f"overbought RSI({cur_rsi:.1f})")
        elif 35 <= cur_rsi <= 45:
            confidence += 8
            reasons.append("RSI in buy zone")
        elif 55 <= cur_rsi <= 65:
            confidence += 8
            reasons.append("RSI in sell zone")
        
        # Volume analysis (basic)
        volumes = [float(c[5]) for c in candles[-5:]]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1]
        
        if current_volume > avg_volume * 1.2:
            confidence += 5
            reasons.append("high volume")
        
        # Generate signals
        entry = current_price
        
        # BUY signal conditions
        if (current_price > cur_ema9 and cur_rsi < 50 and 
            confidence >= 50):
            
            sl = support
            tp = resistance
            
            if entry > sl:
                rr = (tp - entry) / (entry - sl)
                if rr >= rr_min:
                    return {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": min(95, confidence + 10),
                        "reason": "; ".join(reasons),
                        "risk_reward": rr,
                        "indicators": {
                            "rsi": cur_rsi,
                            "ema_9": cur_ema9,
                            "ema_21": cur_ema21
                        }
                    }
        
        # SELL signal conditions
        if (current_price < cur_ema9 and cur_rsi > 50 and 
            confidence >= 50):
            
            sl = resistance
            tp = support
            
            if sl > entry:
                rr = (entry - tp) / (sl - entry)
                if rr >= rr_min:
                    return {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": min(95, confidence + 10),
                        "reason": "; ".join(reasons),
                        "risk_reward": rr,
                        "indicators": {
                            "rsi": cur_rsi,
                            "ema_9": cur_ema9,
                            "ema_21": cur_ema21
                        }
                    }
        
        return {
            "side": "none",
            "confidence": confidence,
            "reason": "; ".join(reasons) if reasons else "no clear signal",
            "indicators": {
                "rsi": cur_rsi,
                "ema_9": cur_ema9,
                "ema_21": cur_ema21
            }
        }
    
    except Exception as e:
        logger.error(f"analyze_trade_logic error: {e}")
        return {"side": "none", "confidence": 0, "reason": f"analysis error: {e}"}

def multi_tf_confirmation(c30m, c1h):
    """Multi-timeframe confirmation with more flexible rules"""
    s30 = analyze_trade_logic(c30m)
    s1h = analyze_trade_logic(c1h)
    
    logger.debug(f"30m: {s30.get('side')} ({s30.get('confidence', 0)}%), 1h: {s1h.get('side')} ({s1h.get('confidence', 0)}%)")
    
    # If both timeframes agree
    if s30.get("side") != "none" and s30["side"] == s1h.get("side"):
        boost = 20
        out = s30.copy()
        out["confidence"] = min(100, s30.get("confidence", 0) + boost)
        out["reason"] = out.get("reason", "") + f"; confirmed by 1h TF"
        return out
    
    # If 30m has strong signal and 1h is neutral
    if (s30.get("side") != "none" and s30.get("confidence", 0) >= 70 and 
        s1h.get("side") == "none"):
        out = s30.copy()
        out["confidence"] = min(100, s30.get("confidence", 0) + 5)
        out["reason"] = out.get("reason", "") + f"; 1h neutral"
        return out
    
    # If 1h has signal and 30m is neutral (but lower confidence)
    if (s1h.get("side") != "none" and s1h.get("confidence", 0) >= 60 and 
        s30.get("side") == "none"):
        out = s1h.copy()
        out["confidence"] = min(100, s1h.get("confidence", 0) - 10)
        out["reason"] = out.get("reason", "") + f"; from 1h TF only"
        return out
    
    return {
        "side": "none",
        "confidence": max(s30.get("confidence", 0), s1h.get("confidence", 0)),
        "reason": f"TF conflict or weak signals: 30m={s30.get('side')}, 1h={s1h.get('side')}"
    }

# ---------------- CHART GENERATION ----------------
def plot_signal_chart(symbol, candles, signal):
    """Create an enhanced chart with indicators"""
    try:
        if not candles:
            return None
        
        # Extract data
        timestamps = [datetime.utcfromtimestamp(int(c[0]) / 1000) for c in candles[-60:]]
        opens = [float(c[1]) for c in candles[-60:]]
        highs = [float(c[2]) for c in candles[-60:]]
        lows = [float(c[3]) for c in candles[-60:]]
        closes = [float(c[4]) for c in candles[-60:]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart with candlesticks (simplified)
        ax1.plot(timestamps, closes, color='blue', linewidth=1, alpha=0.7)
        
        # EMAs
        if len(closes) >= 21:
            ema9_vals = ema(closes, 9)
            ema21_vals = ema(closes, 21)
            
            if ema9_vals and len(ema9_vals) > 0:
                ema9_times = timestamps[-len(ema9_vals):]
                ax1.plot(ema9_times, ema9_vals, color='orange', linewidth=1, label='EMA9', alpha=0.8)
            
            if ema21_vals and len(ema21_vals) > 0:
                ema21_times = timestamps[-len(ema21_vals):]
                ax1.plot(ema21_times, ema21_vals, color='red', linewidth=1, label='EMA21', alpha=0.8)
        
        # Signal levels
        if signal.get("side") != "none":
            ax1.axhline(signal["entry"], color="blue", linestyle="--", alpha=0.8, label=f"Entry: {fmt_price(signal['entry'])}")
            ax1.axhline(signal["sl"], color="red", linestyle="--", alpha=0.8, label=f"SL: {fmt_price(signal['sl'])}")
            ax1.axhline(signal["tp"], color="green", linestyle="--", alpha=0.8, label=f"TP: {fmt_price(signal['tp'])}")
        
        ax1.set_title(f"{symbol} - {signal.get('side', 'NO SIGNAL')} | Confidence: {signal.get('confidence', 0):.1f}%")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # RSI subplot
        if len(closes) >= 14:
            rsi_vals = calculate_rsi(closes)
            if rsi_vals:
                rsi_times = timestamps[-len(rsi_vals):]
                ax2.plot(rsi_times, rsi_vals, color='purple', linewidth=1)
                ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
                ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('RSI')
                ax2.grid(True, alpha=0.3)
        
        # Format dates
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        plt.tight_layout()
        
        # Save to temporary file
        tmp_file = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_file.name, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_file.name
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None

# ---------------- HTTP REQUESTS ----------------
async def fetch_json_with_retry(session, url, max_retries=3):
    """Fetch JSON with retry logic and rate limiting"""
    await rate_limiter.wait_if_needed()
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    wait_time = 60
                    logger.warning(f"Rate limited by API, waiting {wait_time}s")
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

# ---------------- AI ANALYSIS ----------------
async def get_ai_confirmation(symbol, signal_data):
    """Get AI confirmation for signals"""
    if not client:
        return None
    
    try:
        system_prompt = ("You are a professional crypto trader. Analyze the provided market data and "
                        "either CONFIRM or REJECT the trading signal. Be concise.")
        
        user_prompt = f"""
Market: {symbol}
Signal: {signal_data.get('side', 'none')}
Entry: {signal_data.get('entry', 'N/A')}
Confidence: {signal_data.get('confidence', 0)}%
Reason: {signal_data.get('reason', 'N/A')}

Respond with either 'CONFIRM' or 'REJECT' followed by brief reasoning.
"""
        
        loop = asyncio.get_running_loop()
        
        def call_openai():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
        
        response = await loop.run_in_executor(None, call_openai)
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"AI confirmation error: {e}")
        return None

# ---------------- TELEGRAM NOTIFICATIONS ----------------
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, session, text, parse_mode="Markdown"):
        if not self.bot_token or not self.chat_id:
            logger.info(f"[TELEGRAM] {text}")
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text[:4096],  # Telegram limit
                "parse_mode": parse_mode
            }
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Telegram message failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Telegram message error: {e}")
    
    async def send_photo(self, session, caption, photo_path):
        if not self.bot_token or not self.chat_id:
            logger.info(f"[TELEGRAM PHOTO] {caption}")
            try:
                os.unlink(photo_path)
            except:
                pass
            return
        
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('caption', caption[:1024])  # Telegram limit
                data.add_field('photo', photo, filename=os.path.basename(photo_path))
                
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        logger.error(f"Telegram photo failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
        finally:
            try:
                os.unlink(photo_path)
            except:
                pass

telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# ---------------- MARKET SCANNER ----------------
class MarketScanner:
    def __init__(self, symbols):
        self.symbols = symbols
    
    async def scan_all_symbols(self, session):
        """Scan all symbols for trading opportunities"""
        signals = []
        
        # Check rate limits
        recent_signals = db.get_recent_signals_count(hours=1)
        if recent_signals >= MAX_SIGNALS_PER_HOUR:
            logger.warning(f"Signal rate limit reached: {recent_signals}/{MAX_SIGNALS_PER_HOUR}")
            return []
        
        for symbol in self.symbols:
            try:
                signal = await self.analyze_symbol(session, symbol)
                if signal and signal.get("side") != "none":
                    signal["symbol"] = symbol
                    signals.append(signal)
                    logger.info(f"✅ Found signal for {symbol}: {signal.get('side')} ({signal.get('confidence', 0):.1f}%)")
                else:
                    logger.debug(f"❌ No signal for {symbol}")
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return signals[:3]  # Return top 3 signals
    
    async def analyze_symbol(self, session, symbol):
        """Analyze a single symbol"""
        try:
            # Fetch candle data
            c30m_data = await fetch_json_with_retry(session, 
                CANDLE_URL.format(symbol=symbol, interval="30m", limit=100))
            c1h_data = await fetch_json_with_retry(session, 
                CANDLE_URL.format(symbol=symbol, interval="1h", limit=100))
            
            if not c30m_data or not c1h_data:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None
            
            # Filter valid candles
            c30m = [c for c in c30m_data if c and len(c) >= 6]
            c1h = [c for c in c1h_data if c and len(c) >= 6]
            
            if len(c30m) < 30 or len(c1h) < 30:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Multi-timeframe analysis
            signal = multi_tf_confirmation(c30m, c1h)
            
            logger.debug(f"{symbol}: {signal.get('side')} - {signal.get('confidence', 0):.1f}% - {signal.get('reason', '')[:50]}")
            
            # Check if signal meets threshold
            if (signal.get("side") != "none" and 
                signal.get("confidence", 0) >= SIGNAL_CONF_THRESHOLD):
                
                # Get AI confirmation
                ai_confirmation = await get_ai_confirmation(symbol, signal)
                if ai_confirmation and "CONFIRM" in ai_confirmation.upper():
                    signal["confidence"] = min(100, signal.get("confidence", 0) + 5)
                    signal["ai_analysis"] = ai_confirmation
                
                # Generate chart
                chart_path = plot_signal_chart(symbol, c30m, signal)
                if chart_path:
                    signal["chart_path"] = chart_path
                
                return signal
            
            return None
        
        except Exception as e:
            logger.error(f"Symbol analysis error for {symbol}: {e}")
            return None

# ---------------- SIGNAL PROCESSING ----------------
async def process_signal(session, signal):
    """Process and send a trading signal"""
    try:
        symbol = signal.get("symbol", "UNKNOWN")
        side = signal.get("side", "")
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp = signal.get("tp", 0)
        confidence = signal.get("confidence", 0)
        risk_reward = signal.get("risk_reward", 0)
        reason = signal.get("reason", "")
        indicators = signal.get("indicators", {})
        
        # Format message
        message = f"""🎯 *{symbol} {side} SIGNAL*

💰 Entry: `{fmt_price(entry)}`
🛑 Stop Loss: `{fmt_price(sl)}`
🎯 Take Profit: `{fmt_price(tp)}`
📊 Confidence: *{confidence:.1f}%*
📈 Risk/Reward: *{risk_reward:.2f}*

📍 RSI: {indicators.get('rsi', 0):.1f}
📍 EMA9: {fmt_price(indicators.get('ema_9', 0))}
📍 EMA21: {fmt_price(indicators.get('ema_21', 0))}

🔍 Reason: {reason[:200]}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"""
        
        # Send with chart if available
        chart_path = signal.get("chart_path")
        if chart_path and os.path.exists(chart_path):
            await telegram_notifier.send_photo(session, message, chart_path)
        else:
            await telegram_notifier.send_message(session, message)
        
        # Save to database
        db.save_signal({
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "reason": reason
        })
        
        logger.info(f"✅ Signal sent for {symbol} {side}")
        
    except Exception as e:
        logger.error(f"Error processing signal: {e}")

# ---------------- MAIN LOOP ----------------
async def enhanced_trading_loop():
    """Main trading loop"""
    scanner = MarketScanner(SYMBOLS)
    
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Send startup message
        startup_message = f"""🚀 *Enhanced Crypto Trading Bot v4.1 Started*

📊 Monitoring: {len(SYMBOLS)} symbols
⏱️ Scan interval: {POLL_INTERVAL} seconds
🎯 Confidence threshold: {SIGNAL_CONF_THRESHOLD}%
📈 Max signals/hour: {MAX_SIGNALS_PER_HOUR}

Bot is now scanning markets for opportunities..."""
        
        await telegram_notifier.send_message(session, startup_message)
        logger.info("🚀 Trading bot started successfully!")
        
        iteration = 0
        last_stats_time = time.time()
        
        while True:
            iteration += 1
            start_time = time.time()
            
            try:
                logger.info(f"\n{'='*20} SCAN {iteration} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} {'='*20}")
                
                # Scan all symbols
                signals = await scanner.scan_all_symbols(session)
                
                scan_duration = time.time() - start_time
                logger.info(f"Scan completed in {scan_duration:.1f}s - Found {len(signals)} signals")
                
                # Process signals
                for signal in signals:
                    try:
                        await process_signal(session, signal)
                        # Small delay between signals
                        await asyncio.sleep(2)
                    except Exception as e:
                        logger.error(f"Error processing signal: {e}")
                
                # Send periodic stats (every 4 hours)
                if time.time() - last_stats_time > 14400:
                    stats_message = f"""📊 *Bot Statistics Update*

🔄 Scans completed: {iteration}
⏰ Runtime: {(time.time() - start_time)/3600:.1f} hours
💾 Recent signals: {db.get_recent_signals_count(hours=24)} (24h)

Bot is running normally..."""
                    
                    await telegram_notifier.send_message(session, stats_message)
                    last_stats_time = time.time()
                
                # Dynamic sleep time
                if signals:
                    # If we found signals, scan more frequently
                    sleep_time = max(300, POLL_INTERVAL // 3)  # Minimum 5 minutes
                    logger.info(f"Found {len(signals)} signals - next scan in {sleep_time}s")
                else:
                    # No signals, use normal interval
                    sleep_time = POLL_INTERVAL
                    logger.info(f"No signals found - next scan in {sleep_time}s")
                
                await asyncio.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                logger.error(traceback.format_exc())
                
                # Send error notification
                error_message = f"🚨 *Bot Error*\n\n`{str(e)[:500]}`\n\nBot will continue in 60 seconds..."
                await telegram_notifier.send_message(session, error_message)
                
                # Wait before retrying
                await asyncio.sleep(60)

# ---------------- CLI HELPERS ----------------
async def test_single_symbol(symbol):
    """Test analysis for a single symbol"""
    scanner = MarketScanner([symbol])
    
    async with aiohttp.ClientSession() as session:
        logger.info(f"Testing analysis for {symbol}...")
        
        try:
            signal = await scanner.analyze_symbol(session, symbol)
            
            if signal:
                print(f"\n✅ Signal found for {symbol}:")
                print(f"Side: {signal.get('side')}")
                print(f"Confidence: {signal.get('confidence', 0):.1f}%")
                print(f"Entry: {fmt_price(signal.get('entry', 0))}")
                print(f"SL: {fmt_price(signal.get('sl', 0))}")
                print(f"TP: {fmt_price(signal.get('tp', 0))}")
                print(f"R/R: {signal.get('risk_reward', 0):.2f}")
                print(f"Reason: {signal.get('reason', '')}")
                
                # Generate chart
                chart_path = signal.get('chart_path')
                if chart_path:
                    print(f"Chart saved to: {chart_path}")
            else:
                print(f"\n❌ No signal found for {symbol}")
        
        except Exception as e:
            print(f"\n🚨 Error testing {symbol}: {e}")

async def run_single_scan():
    """Run a single scan of all symbols"""
    scanner = MarketScanner(SYMBOLS)
    
    async with aiohttp.ClientSession() as session:
        logger.info("Running single scan of all symbols...")
        
        try:
            signals = await scanner.scan_all_symbols(session)
            
            print(f"\n📊 Scan Results: {len(signals)} signals found")
            print("-" * 60)
            
            for i, signal in enumerate(signals, 1):
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', '')
                confidence = signal.get('confidence', 0)
                reason = signal.get('reason', '')
                
                print(f"{i}. {symbol} - {side} ({confidence:.1f}%)")
                print(f"   Reason: {reason[:60]}...")
                print()
        
        except Exception as e:
            print(f"\n🚨 Error during scan: {e}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "test" and len(sys.argv) > 2:
                # Test single symbol: python main.py test BTCUSDT
                symbol = sys.argv[2]
                asyncio.run(test_single_symbol(symbol))
            
            elif command == "scan":
                # Run single scan: python main.py scan
                asyncio.run(run_single_scan())
            
            else:
                print("Usage:")
                print("  python main.py              # Run main bot")
                print("  python main.py test SYMBOL  # Test single symbol")
                print("  python main.py scan          # Run single scan")
        
        else:
            # Run main bot
            logger.info("Starting Enhanced Crypto Trading Bot v4.1")
            asyncio.run(enhanced_trading_loop())
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
