import os
import asyncio
import logging
from datetime import datetime, timedelta
import redis
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import mplfinance as mpf
import io
from PIL import Image

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Constants
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
SYMBOLS = ['BTC-PERPETUAL', 'ETH-PERPETUAL']
TIMEFRAMES = ['30', '60', '240']  # 30m, 1h, 4h in minutes
MAX_TRADES_PER_DAY = 8

class DeribitClient:
    """Fetch data from Deribit public API"""
    
    @staticmethod
    def get_candles(symbol: str, timeframe: str, count: int = 150) -> pd.DataFrame:
        """Fetch OHLCV data"""
        url = f"{DERIBIT_BASE}/get_tradingview_chart_data"
        params = {
            'instrument_name': symbol,
            'resolution': timeframe,
            'start_timestamp': int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
            'end_timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['result']['status'] == 'ok':
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['result']['ticks'], unit='ms'),
                    'open': data['result']['open'],
                    'high': data['result']['high'],
                    'low': data['result']['low'],
                    'close': data['result']['close'],
                    'volume': data['result']['volume']
                })
                df.set_index('timestamp', inplace=True)
                return df.tail(count)
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
        return pd.DataFrame()
    
    @staticmethod
    def get_order_book(symbol: str, depth: int = 10) -> Dict:
        """Fetch order book for OI analysis"""
        url = f"{DERIBIT_BASE}/get_order_book"
        params = {'instrument_name': symbol, 'depth': depth}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if 'result' in data:
                return {
                    'open_interest': data['result'].get('open_interest', 0),
                    'volume_24h': data['result'].get('stats', {}).get('volume', 0),
                    'mark_price': data['result'].get('mark_price', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
        return {}

class TechnicalAnalyzer:
    """Technical analysis functions"""
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, period: int = 5) -> Tuple[List, List]:
        """Identify swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(period, len(df) - period):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                swing_highs.append({'price': df['high'].iloc[i], 'index': i})
            
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                swing_lows.append({'price': df['low'].iloc[i], 'index': i})
        
        return swing_highs[-7:], swing_lows[-7:]  # Last 7 swing points
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        
        # Doji
        if body < (last['high'] - last['low']) * 0.1:
            patterns.append("Doji")
        
        # Hammer
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns.append("Hammer")
        
        # Shooting Star
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            patterns.append("Shooting Star")
        
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and 
            last['close'] > last['open'] and 
            last['close'] > prev['open'] and 
            last['open'] < prev['close']):
            patterns.append("Bullish Engulfing")
        
        # Bearish Engulfing
        if (prev['close'] > prev['open'] and 
            last['close'] < last['open'] and 
            last['close'] < prev['open'] and 
            last['open'] > prev['close']):
            patterns.append("Bearish Engulfing")
        
        # Three White Soldiers
        if (last['close'] > last['open'] and 
            prev['close'] > prev['open'] and 
            prev2['close'] > prev2['open'] and
            last['close'] > prev['close'] > prev2['close']):
            patterns.append("Three White Soldiers")
        
        return patterns
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame) -> Dict:
        """Calculate high volume nodes"""
        price_range = df['high'].max() - df['low'].min()
        bins = 20
        bin_size = price_range / bins
        
        volume_by_price = {}
        for _, row in df.iterrows():
            price_bin = int((row['close'] - df['low'].min()) / bin_size)
            volume_by_price[price_bin] = volume_by_price.get(price_bin, 0) + row['volume']
        
        poc_bin = max(volume_by_price, key=volume_by_price.get)
        poc_price = df['low'].min() + (poc_bin * bin_size)
        
        return {
            'poc': poc_price,
            'high_volume_nodes': sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)[:3]
        }

class OITracker:
    """Track OI changes using Redis"""
    
    @staticmethod
    def store_oi(symbol: str, oi_data: Dict):
        """Store current OI in Redis with timestamp"""
        key = f"oi:{symbol}:{int(datetime.now().timestamp())}"
        redis_client.setex(key, 7200, json.dumps(oi_data))  # 2hr expiry
    
    @staticmethod
    def get_oi_history(symbol: str, hours: int = 2) -> List[Dict]:
        """Get OI history from Redis"""
        cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
        pattern = f"oi:{symbol}:*"
        
        history = []
        for key in redis_client.scan_iter(match=pattern):
            timestamp = int(key.split(':')[-1])
            if timestamp >= cutoff:
                data = json.loads(redis_client.get(key))
                data['timestamp'] = timestamp
                history.append(data)
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    @staticmethod
    def analyze_oi_trend(symbol: str) -> Dict:
        """Analyze OI trend over last 2 hours"""
        history = OITracker.get_oi_history(symbol, hours=2)
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        
        current_oi = history[-1]['open_interest']
        old_oi = history[0]['open_interest']
        change = ((current_oi - old_oi) / old_oi * 100) if old_oi > 0 else 0
        
        if change > 5:
            trend = 'strongly_increasing'
        elif change > 2:
            trend = 'increasing'
        elif change < -5:
            trend = 'strongly_decreasing'
        elif change < -2:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': round(change, 2),
            'current_oi': current_oi,
            'previous_oi': old_oi
        }

class TradeAnalyzer:
    """Main trade analysis engine"""
    
    @staticmethod
    def analyze_setup(symbol: str) -> Dict:
        """Comprehensive trade analysis"""
        # Fetch multi-timeframe data
        df_30m = DeribitClient.get_candles(symbol, '30')
        df_1h = DeribitClient.get_candles(symbol, '60')
        df_4h = DeribitClient.get_candles(symbol, '240')
        
        if df_30m.empty or df_1h.empty or df_4h.empty:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        # Get OI data
        oi_data = DeribitClient.get_order_book(symbol)
        OITracker.store_oi(symbol, oi_data)
        oi_trend = OITracker.analyze_oi_trend(symbol)
        
        # Technical analysis
        swing_highs_30m, swing_lows_30m = TechnicalAnalyzer.find_swing_points(df_30m)
        swing_highs_1h, swing_lows_1h = TechnicalAnalyzer.find_swing_points(df_1h)
        swing_highs_4h, swing_lows_4h = TechnicalAnalyzer.find_swing_points(df_4h)
        
        patterns = TechnicalAnalyzer.detect_patterns(df_30m)
        volume_profile = TechnicalAnalyzer.calculate_volume_profile(df_30m)
        
        current_price = df_30m['close'].iloc[-1]
        avg_volume = df_30m['volume'].tail(20).mean()
        current_volume = df_30m['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Determine trade setup
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'patterns': patterns,
            'volume_ratio': round(volume_ratio, 2),
            'oi_trend': oi_trend,
            'swing_highs_30m': swing_highs_30m,
            'swing_lows_30m': swing_lows_30m,
            'swing_highs_1h': swing_highs_1h,
            'swing_lows_1h': swing_lows_1h,
            'resistance_4h': swing_highs_4h[-1]['price'] if swing_highs_4h else None,
            'support_4h': swing_lows_4h[-1]['price'] if swing_lows_4h else None,
            'volume_profile': volume_profile,
            'valid': False
        }
        
        return analysis
    
    @staticmethod
    def get_ai_analysis(analysis: Dict) -> str:
        """Get GPT-4o mini analysis"""
        prompt = f"""Analyze this crypto trading setup:

Symbol: {analysis['symbol']}
Price: ${analysis['current_price']}
Timeframe: 30min

Patterns Detected: {', '.join(analysis['patterns']) if analysis['patterns'] else 'None'}
Volume Ratio: {analysis['volume_ratio']}x average
OI Trend: {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']}%)

Support/Resistance:
- 30m swing low: ${analysis['swing_lows_30m'][-1]['price'] if analysis['swing_lows_30m'] else 'N/A'}
- 30m swing high: ${analysis['swing_highs_30m'][-1]['price'] if analysis['swing_highs_30m'] else 'N/A'}
- 4h support: ${analysis['support_4h']}
- 4h resistance: ${analysis['resistance_4h']}

Rules to follow:
1. Valid breakout/breakdown requires body close beyond S/R
2. Wicks should be < 30% of candle (clean break)
3. Volume must be > 1.5x average
4. OI should be increasing for valid moves
5. Multi-TF alignment needed

Determine:
- Is this a valid LONG/SHORT setup or NO TRADE?
- Entry, Stop Loss (recent swing), Target (R:R 1:2 minimum)
- Brief explanation (2-3 lines max)

Format:
SIGNAL: [LONG/SHORT/NO_TRADE]
ENTRY: $X
SL: $X
TARGET: $X
REASON: [brief explanation]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert crypto trader specializing in breakout/breakdown strategies with strict risk management."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT API error: {e}")
            return "AI analysis failed"

class TradingBot:
    """Main bot logic"""
    
    def __init__(self):
        self.trade_count_today = 0
        self.last_reset = datetime.now().date()
    
    def reset_daily_counter(self):
        """Reset trade counter at midnight"""
        if datetime.now().date() > self.last_reset:
            self.trade_count_today = 0
            self.last_reset = datetime.now().date()
    
    async def scan_markets(self, context: ContextTypes.DEFAULT_TYPE):
        """Scan all symbols for setups"""
        self.reset_daily_counter()
        
        if self.trade_count_today >= MAX_TRADES_PER_DAY:
            logger.info(f"Daily limit reached: {self.trade_count_today}/{MAX_TRADES_PER_DAY}")
            return
        
        for symbol in SYMBOLS:
            try:
                analysis = TradeAnalyzer.analyze_setup(symbol)
                
                # Basic filters
                if (analysis.get('volume_ratio', 0) < 1.5 or 
                    not analysis.get('patterns')):
                    continue
                
                # Get AI analysis
                ai_result = TradeAnalyzer.get_ai_analysis(analysis)
                
                # Check if valid signal
                if 'LONG' in ai_result or 'SHORT' in ai_result:
                    if 'NO_TRADE' not in ai_result:
                        self.trade_count_today += 1
                        await self.send_alert(context, symbol, analysis, ai_result)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
            
            await asyncio.sleep(2)  # Rate limit
    
    async def send_alert(self, context: ContextTypes.DEFAULT_TYPE, symbol: str, analysis: Dict, ai_result: str):
        """Send trade alert to Telegram"""
        message = f"""🔔 **{symbol} - TRADE SETUP**

📊 **Analysis:**
Price: ${analysis['current_price']}
Patterns: {', '.join(analysis['patterns'])}
Volume: {analysis['volume_ratio']}x avg
OI: {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']}%)

{ai_result}

⚠️ Trade #{self.trade_count_today}/{MAX_TRADES_PER_DAY} today
"""
        
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='Markdown'
        )

# Bot commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    await update.message.reply_text(
        "🤖 Trading Bot Active!\n\n"
        "Scanning BTC & ETH every 30 mins\n"
        "Max 8 trades/day\n\n"
        "Commands:\n"
        "/status - Check bot status\n"
        "/analyze BTC - Manual analysis"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status command"""
    bot = context.bot_data.get('trading_bot')
    if bot:
        await update.message.reply_text(
            f"📊 Status:\n"
            f"Trades today: {bot.trade_count_today}/{MAX_TRADES_PER_DAY}\n"
            f"Next scan: Every 30 mins"
        )

def main():
    """Main function"""
    # Initialize bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    trading_bot = TradingBot()
    application.bot_data['trading_bot'] = trading_bot
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    
    # Schedule market scans every 30 mins
    job_queue = application.job_queue
    job_queue.run_repeating(
        trading_bot.scan_markets,
        interval=1800,  # 30 mins
        first=10
    )
    
    # Start bot
    logger.info("Bot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
