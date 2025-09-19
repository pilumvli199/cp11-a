#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v5.0 (1-request-per-second scanning)
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

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(10, int(os.getenv("POLL_INTERVAL", 1800)))  # default large but can be reduced for testing
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
            confidence REAL NOT NULL, reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'active')""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON signals(symbol, timestamp);")
        conn.commit(); conn.close()
    def save_signal(self, signal_data: Dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""INSERT INTO signals (symbol, side, entry_price, sl_price, tp_price, confidence, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                        signal_data['symbol'], signal_data['side'], signal_data['entry'],
                        signal_data['sl'], signal_data['tp'], signal_data['confidence'], signal_data['reason']))
        conn.commit(); conn.close()
    def get_recent_signals_count(self, hours=1) -> int:
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now() - timedelta(hours=hours)
        cur = conn.execute("SELECT COUNT(*) FROM signals WHERE timestamp > ? AND status = 'active'", (cutoff,))
        cnt = cur.fetchone()[0]; conn.close(); return cnt

db = SignalDatabase()

# ---------------- RATE LIMITER: 1 request per second ----------------
class RateLimiter:
    """
    Concurrency-safe sliding-window rate limiter.
    Configured for 1 request / 1 second by default.
    """
    def __init__(self, max_requests=1, window_seconds=1.0, log_suppress_interval=1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
        self.last_log_time = 0.0
        self.log_suppress_interval = log_suppress_interval

    async def wait_if_needed(self):
        async with self._lock:
            now = time.time()
            # drop old
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
            if now - self.last_log_time > self.log_suppress_interval:
                logger.info(f"Rate limit reached (local), waiting {sleep_time:.2f}s")
                self.last_log_time = now
        await asyncio.sleep(sleep_time)
        async with self._lock:
            now2 = time.time()
            self.requests = [t for t in self.requests if now2 - t < self.window_seconds]
            self.requests.append(now2)

# instantiate for 1 req/sec
rate_limiter = RateLimiter(max_requests=1, window_seconds=1.0, log_suppress_interval=1.0)

# Global remote backoff when 429 happens
remote_backoff_until = 0.0
REMOTE_BACKOFF_DEFAULT = 60  # seconds

# ---------------- UTILITIES / INDICATORS ----------------
def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    if abs(p) < 0.001: return f"{p:.8f}"
    if abs(p) < 1: return f"{p:.6f}"
    if abs(p) < 100: return f"{p:.4f}"
    return f"{p:.2f}"

def calculate_rsi(closes, period=RSI_PERIOD):
    if len(closes) < period + 1: return []
    deltas = [closes[i]-closes[i-1] for i in range(1,len(closes))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period])/period
    avg_loss = sum(losses[:period])/period
    rsi_vals=[]
    for i in range(period, len(closes)):
        if avg_loss == 0: rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100/(1+rs))
        rsi_vals.append(rsi)
        if i < len(deltas):
            gain = gains[i] if i < len(gains) else 0
            loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain*(period-1)+gain)/period
            avg_loss = (avg_loss*(period-1)+loss)/period
    return rsi_vals

def ema(values, period):
    if len(values) < period: return []
    k = 2/(period+1)
    ema_vals = [None]*(period-1)
    sma = sum(values[:period])/period
    ema_vals.append(sma)
    for i in range(period, len(values)):
        ema_vals.append(values[i]*k + ema_vals[-1]*(1-k))
    return ema_vals

def calculate_macd(closes):
    if len(closes) < MACD_SLOW: return {"macd":[], "signal":[], "histogram":[]}
    fast = ema(closes, MACD_FAST); slow = ema(closes, MACD_SLOW)
    macd_line=[]
    for i in range(len(closes)):
        f = fast[i] if i < len(fast) else None
        s = slow[i] if i < len(slow) else None
        if f is not None and s is not None:
            macd_line.append(f-s)
    signal_line = ema(macd_line, MACD_SIGNAL) if macd_line else []
    signal_clean = [x for x in signal_line if x is not None]
    hist = []
    for i in range(min(len(macd_line), len(signal_clean))):
        hist.append(macd_line[i]-signal_clean[i])
    return {"macd": macd_line, "signal": signal_clean, "histogram": hist}

def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    if len(closes) < lookback: lookback = len(closes)
    pts = closes[-lookback:] + highs[-lookback:] + lows[-lookback:]
    levels=[]
    for p in pts:
        if p is None or p == 0: continue
        found=False
        for lv in levels:
            try:
                if abs((lv["price"]-p)/p) < binsize:
                    lv["count"] += 1
                    lv["price"] = (lv["price"]*(lv["count"]-1) + p)/lv["count"]
                    found=True; break
            except:
                continue
        if not found:
            levels.append({"price": p, "count": 1})
    levels.sort(key=lambda x: -x["count"])
    return [lv["price"] for lv in levels if lv["count"] >= 3][:5]

# ---------------- ANALYSIS ----------------
def analyze_trade_logic(candles, rr_min=1.5):
    try:
        if not candles or len(candles) < 50:
            return {"side":"none","confidence":0,"reason":"insufficient data"}
        cleaned=[]
        for c in candles:
            try:
                if not c or len(c) < 5: continue
                o=float(c[1]); h=float(c[2]); l=float(c[3]); cl=float(c[4])
                cleaned.append([o,h,l,cl])
            except:
                continue
        if len(cleaned) < 50:
            return {"side":"none","confidence":0,"reason":"insufficient valid candles after cleaning"}
        closes=[x[3] for x in cleaned]; highs=[x[1] for x in cleaned]; lows=[x[2] for x in cleaned]
        current_price = closes[-1]
        ema9 = ema(closes,9); ema21 = ema(closes,21); ema50 = ema(closes,50)
        rsi_vals = calculate_rsi(closes); macd = calculate_macd(closes)
        if not ema9 or not ema21 or not rsi_vals:
            return {"side":"none","confidence":0,"reason":"indicator calc failed"}
        cur_ema9 = ema9[-1] if ema9 and ema9[-1] is not None else None
        cur_ema21 = ema21[-1] if ema21 and ema21[-1] is not None else None
        cur_ema50 = (ema50[-1] if ema50 and ema50[-1] is not None else cur_ema21)
        cur_rsi = rsi_vals[-1] if rsi_vals else None
        if cur_ema9 is None or cur_ema21 is None or cur_rsi is None:
            return {"side":"none","confidence":0,"reason":"insufficient indicator values (None found)"}
        levels = horizontal_levels(closes, highs, lows)
        support = max([lv for lv in levels if lv < current_price], default=None) if levels else None
        resistance = min([lv for lv in levels if lv > current_price], default=None) if levels else None
        confidence = 50; reasons=[]
        try:
            if current_price > cur_ema9 > cur_ema21 > cur_ema50:
                confidence += 15; reasons.append("strong uptrend (EMAs)")
            elif current_price < cur_ema9 < cur_ema21 < cur_ema50:
                confidence += 15; reasons.append("strong downtrend (EMAs)")
            elif current_price > cur_ema9 > cur_ema21:
                confidence += 10; reasons.append("uptrend short-term")
            elif current_price < cur_ema9 < cur_ema21:
                confidence += 10; reasons.append("downtrend short-term")
            else:
                confidence -= 5; reasons.append("mixed EMA")
        except:
            reasons.append("ema ordering check failed")
        if cur_rsi is not None:
            if cur_rsi > 70: confidence += 5; reasons.append(f"overbought RSI({cur_rsi:.1f})")
            elif cur_rsi < 30: confidence += 5; reasons.append(f"oversold RSI({cur_rsi:.1f})")
        try:
            if macd["macd"] and macd["signal"]:
                mcur = macd["macd"][-1]; scur = macd["signal"][-1]
                if mcur > scur and len(macd["macd"])>1:
                    prev_m = macd["macd"][-2]
                    if prev_m <= scur: confidence += 10; reasons.append("MACD bullish crossover")
                elif mcur < scur and len(macd["macd"])>1:
                    prev_m = macd["macd"][-2]
                    if prev_m >= scur: confidence += 10; reasons.append("MACD bearish crossover")
        except:
            reasons.append("macd check failed")
        # Construct candidates if levels exist and confidence satisfy
        if support is not None and resistance is not None and confidence >= 60:
            entry = current_price
            sl = support * 0.995
            tp = resistance * 0.995
            denom = (entry - sl)
            if denom > 0:
                rr = (tp - entry)/denom
                if rr >= rr_min:
                    return {"side":"BUY","entry":entry,"sl":sl,"tp":tp,"confidence":min(95,confidence+10),"reason":"; ".join(reasons),"risk_reward":rr,"indicators":{"rsi":cur_rsi,"ema_9":cur_ema9,"ema_21":cur_ema21}}
            sl_s = resistance * 1.005
            tp_s = support * 1.005
            denom2 = (sl_s - entry)
            if denom2 > 0:
                rr2 = (entry - tp_s)/denom2
                if rr2 >= rr_min:
                    return {"side":"SELL","entry":entry,"sl":sl_s,"tp":tp_s,"confidence":min(95,confidence+10),"reason":"; ".join(reasons),"risk_reward":rr2,"indicators":{"rsi":cur_rsi,"ema_9":cur_ema9,"ema_21":cur_ema21}}
        return {"side":"none","confidence":confidence,"reason":"; ".join(reasons),"indicators":{"rsi":cur_rsi or 50,"ema_9":cur_ema9 or current_price,"ema_21":cur_ema21 or current_price}}
    except Exception as e:
        logger.error(f"analyze_trade_logic fatal: {e}"); logger.error(traceback.format_exc())
        return {"side":"none","confidence":0,"reason":f"internal error: {e}"}

def multi_tf_confirmation(c30m, c1h, c4h=None):
    s30 = analyze_trade_logic(c30m); s1 = analyze_trade_logic(c1h)
    if s30.get("side") == "none" or s1.get("side") == "none":
        return {"side":"none","confidence":0,"reason":"no clear signals on timeframes"}
    if s30["side"] == s1["side"]:
        boost = 20
        if c4h:
            s4 = analyze_trade_logic(c4h)
            if s4.get("side") == s30["side"]: boost += 10
        out = s30.copy(); out["confidence"] = min(100, s30.get("confidence",0) + boost)
        out["reason"] = out.get("reason","") + f"; aligned with 1h TF (conf:{s1.get('confidence',0)}%)"
        return out
    return {"side":"none","confidence":0,"reason":f"TF conflict:30m={s30.get('side')},1h={s1.get('side')}"}

# ---------------- CHART (simple) ----------------
def plot_simple_chart(symbol, candles, signal):
    try:
        good=[]
        for row in candles[-60:]:
            try:
                t=int(row[0]); c=float(row[4]); good.append((t,c))
            except: continue
        if not good:
            tmp=NamedTemporaryFile(delete=False,suffix=".png"); plt.figure(); plt.text(0.5,0.5,"No data",ha='center'); plt.savefig(tmp.name); plt.close(); return tmp.name
        dates=[datetime.utcfromtimestamp(int(t)/1000) for t,c in good]; closes=[c for t,c in good]
        plt.figure(figsize=(10,5)); plt.plot(dates,closes); plt.title(f"{symbol} - {signal.get('side','none')}"); plt.grid(True)
        if signal.get("side")!="none":
            try: plt.axhline(signal["entry"],linestyle='--'); plt.axhline(signal["sl"],linestyle='--'); plt.axhline(signal["tp"],linestyle='--')
            except: pass
        tmp=NamedTemporaryFile(delete=False,suffix=".png"); plt.savefig(tmp.name,bbox_inches='tight'); plt.close(); return tmp.name
    except Exception as e:
        logger.error(f"plot_simple_chart failed: {e}"); tmp=NamedTemporaryFile(delete=False,suffix=".png"); plt.figure(); plt.text(0.5,0.5,"Chart error",ha='center'); plt.savefig(tmp.name); plt.close(); return tmp.name

# ---------------- HTTP SESSION & FETCH WITH GLOBAL BACKOFF ----------------
@asynccontextmanager
async def get_session():
    # small connector limit; we rely on per-second scheduling
    connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300, use_dns_cache=True)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        yield session

async def fetch_json_with_retry(session, url, max_retries=3):
    global remote_backoff_until
    now = time.time()
    if now < remote_backoff_until:
        wait = remote_backoff_until - now
        logger.info(f"Global remote backoff active, sleeping {wait:.1f}s before request")
        await asyncio.sleep(wait)
    await rate_limiter.wait_if_needed()  # enforce 1 req/sec
    for attempt in range(max_retries):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    # remote rate-limit: set global backoff and return None
                    logger.warning(f"Remote returned 429 for {url} (attempt {attempt+1}). Setting global backoff {REMOTE_BACKOFF_DEFAULT}s.")
                    remote_backoff_until = time.time() + REMOTE_BACKOFF_DEFAULT
                    await asyncio.sleep(REMOTE_BACKOFF_DEFAULT)
                    return None
                else:
                    logger.warning(f"HTTP {resp.status} for {url}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url} (attempt {attempt+1})")
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
        await asyncio.sleep(min(30, 1 * (2 ** attempt)))
    return None

# ---------------- AI confirmation (minimal) ----------------
async def get_enhanced_ai_analysis(symbol, candles, signal):
    if not client: return None
    try:
        closes=[]
        for c in candles[-20:]:
            try: closes.append(float(c[4]))
            except: continue
        if not closes: return None
        current_price = closes[-1]
        market_summary = {"symbol":symbol,"current_price":current_price,"local_signal": {"side":signal.get("side"), "confidence":signal.get("confidence")}}
        system_prompt = "You are an expert trader. Provide short confirm/reject of the local signal (CONF in %). If no signal, reply NO_SIGNAL."
        user_prompt = f"Market Data:\n{json.dumps(market_summary)}\nPlease respond with short confirmation or NO_SIGNAL."
        loop = asyncio.get_running_loop()
        def call_openai():
            return client.chat.completions.create(model=OPENAI_MODEL, messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], max_tokens=200, temperature=0.2)
        resp = await loop.run_in_executor(None, call_openai)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI error: {e}"); return None

# ---------------- TELEGRAM ----------------
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token; self.chat_id = chat_id; self.base = f"https://api.telegram.org/bot{bot_token}"
    async def send_message(self, session, text, parse_mode="Markdown"):
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram configured, msg: {text[:120]}"); return
        try:
            async with session.post(f"{self.base}/sendMessage", json={"chat_id":self.chat_id,"text":text[:4096],"parse_mode":parse_mode}) as r:
                if r.status != 200: logger.error(f"Telegram send failed: {r.status}")
        except Exception as e: logger.error(f"Telegram msg err: {e}")
    async def send_photo(self, session, caption, path):
        if not self.bot_token or not self.chat_id:
            logger.info(f"No Telegram configured, caption: {caption[:120]}")
            try: os.unlink(path)
            except: pass
            return
        try:
            with open(path,'rb') as f:
                data = aiohttp.FormData(); data.add_field("chat_id", self.chat_id); data.add_field("caption", caption[:1024]); data.add_field("photo", f, filename=os.path.basename(path))
                async with session.post(f"{self.base}/sendPhoto", data=data) as r:
                    if r.status != 200: logger.error(f"Telegram photo failed: {r.status}")
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
        finally:
            try: os.unlink(path)
            except: pass

telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# ---------------- MARKET SCANNER (1 sec per symbol) ----------------
class MarketScanner:
    def __init__(self, symbols):
        self.symbols = symbols
    async def scan_all_symbols(self, session):
        signals=[]
        recent = db.get_recent_signals_count(hours=1)
        if recent >= MAX_SIGNALS_PER_HOUR:
            logger.warning(f"Signal rate limit reached {recent}/{MAX_SIGNALS_PER_HOUR} per hour"); return []
        tasks=[]
        # schedule tasks with 1-second stagger so overall ~N seconds for N symbols
        for symbol in self.symbols:
            tasks.append(asyncio.create_task(self.analyze_symbol(session, symbol)))
            await asyncio.sleep(1.0)  # 1 request per second pacing
        sem = asyncio.Semaphore(1)  # only one symbol processed at a time
        async def limited(t):
            async with sem:
                return await t
        results = await asyncio.gather(*[limited(t) for t in tasks], return_exceptions=True)
        for i,res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Err analyzing {self.symbols[i]}: {res}"); continue
            if res and res.get("side") != "none":
                res["symbol"] = self.symbols[i]; signals.append(res)
        signals.sort(key=lambda x: x.get("confidence",0), reverse=True)
        return signals[:3]
    async def analyze_symbol(self, session, symbol):
        try:
            c30 = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="30m", limit=100))
            c1  = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="1h", limit=100))
            c4  = await fetch_json_with_retry(session, CANDLE_URL.format(symbol=symbol, interval="4h", limit=50))
            if not c30 or not c1:
                logger.warning(f"Failed fetch for {symbol}"); return None
            c30m = [c for c in c30 if c and len(c) >= 5]
            c1h  = [c for c in c1 if c and len(c) >= 5]
            c4h  = [c for c in c4 if c and len(c) >= 5] if c4 else None
            signal = multi_tf_confirmation(c30m, c1h, c4h)
            if signal.get("side") != "none" and signal.get("confidence",0) >= SIGNAL_CONF_THRESHOLD:
                ai = await get_enhanced_ai_analysis(symbol, c30m, signal)
                if ai and "NO_SIGNAL" not in ai.upper():
                    signal["confidence"] = min(100, signal.get("confidence",0) + 5); signal["ai_analysis"] = ai
                chart = plot_simple_chart(symbol, c30m, signal); signal["chart_path"] = chart
                return signal
            return None
        except Exception as e:
            logger.error(f"Symbol analysis error for {symbol}: {e}"); logger.error(traceback.format_exc()); return None

# ---------------- PROCESS SIGNAL & MAIN LOOP ----------------
async def process_signal(session, signal):
    sym = signal.get("symbol","UNKNOWN"); rr = signal.get("risk_reward",0); ind = signal.get("indicators",{})
    msg = (f"🎯 *{sym} {signal.get('side','')} SIGNAL*\n\n"
           f"💰 Entry: `{fmt_price(signal.get('entry','N/A'))}`\n"
           f"🛑 SL: `{fmt_price(signal.get('sl','N/A'))}`\n"
           f"🎯 TP: `{fmt_price(signal.get('tp','N/A'))}`\n"
           f"📊 Conf: *{signal.get('confidence',0):.1f}%* • R/R: *{rr:.2f}*\n\n"
           f"📈 RSI: {ind.get('rsi',0):.1f} • EMA9: {fmt_price(ind.get('ema_9',0))} • EMA21: {fmt_price(ind.get('ema_21',0))}\n\n"
           f"{signal.get('reason','')[:300]}\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    try:
        chart = signal.get("chart_path")
        if chart and os.path.exists(chart):
            await telegram.send_photo(session, msg, chart)
        else:
            await telegram.send_message(session, msg)
        if all(k in signal for k in ("symbol","side","entry","sl","tp","confidence","reason")):
            db.save_signal({"symbol":signal["symbol"],"side":signal["side"],"entry":signal["entry"],"sl":signal["sl"],"tp":signal["tp"],"confidence":signal["confidence"],"reason":signal["reason"]})
        logger.info(f"✅ Sent signal {sym} {signal.get('side')} (Conf {signal.get('confidence')})")
    except Exception as e:
        logger.error(f"Failed to send/process signal {sym}: {e}"); logger.error(traceback.format_exc())

async def enhanced_trading_loop():
    scanner = MarketScanner(SYMBOLS)
    async with get_session() as session:
        startup = (f"🚀 Enhanced Trading Bot v5.0 Started\nSymbols: {len(SYMBOLS)} • Per-second pacing: 1 symbol/sec\nScan interval: {POLL_INTERVAL}s • Conf≥{SIGNAL_CONF_THRESHOLD}%")
        logger.info("Trading bot started successfully!"); await telegram.send_message(session, startup)
        iteration = 0; last_stats = time.time()
        while True:
            iteration += 1; start = time.time()
            try:
                logger.info(f"\n=== SCAN {iteration} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} ===")
                signals = await scanner.scan_all_symbols(session)
                logger.info(f"Scan done in {time.time()-start:.1f}s — signals: {len(signals)}")
                for s in signals:
                    try:
                        await process_signal(session, s); await asyncio.sleep(2)
                    except Exception as e:
                        logger.error(f"process signal err: {e}")
                if time.time() - last_stats > 14400:
                    await telegram.send_message(session, f"Bot stats: scanned {iteration} cycles"); last_stats = time.time()
                sleep_time = POLL_INTERVAL if not signals else max(300, POLL_INTERVAL//2)
                logger.info(f"Sleeping {sleep_time}s until next scan..."); await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Main loop error: {e}"); logger.error(traceback.format_exc()); await telegram.send_message(session, f"Bot error: {str(e)[:200]}"); await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        logger.info("Starting Enhanced Crypto Trading Bot v5.0")
        asyncio.run(enhanced_trading_loop())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal: {e}"); logger.error(traceback.format_exc())
