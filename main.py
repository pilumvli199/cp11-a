import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt
from telegram import Bot
from telegram.error import TelegramError
import logging
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

# ======================== CONFIGURATION ========================
DERIBIT_API_URL = "https://www.deribit.com/api/v2/public"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

TIMEFRAME = "30"  # 30 minutes
CANDLES_FOR_ANALYSIS = 500
CANDLES_FOR_CHART = 200
SR_ZONE_PERCENT = 0.004  # 0.4% zone width
MIN_TOUCHES_NORMAL = 2  # Normal S/R needs 2-3 touches
MIN_VOLUME_SPIKE = 1.5  # 1.5x average volume for confirmation

# Top 20 coins for analysis (Deribit perpetual format)
TOP_COINS = [
    "BTC-PERPETUAL",      # Bitcoin
    "ETH-PERPETUAL",      # Ethereum
    "SOL-PERPETUAL",      # Solana
    "XRP-PERPETUAL",      # Ripple
    "BNB-PERPETUAL",      # Binance Coin
    "ADA-PERPETUAL",      # Cardano
    "AVAX-PERPETUAL",     # Avalanche
    "DOGE-PERPETUAL",     # Dogecoin
    "MATIC-PERPETUAL",    # Polygon
    "LTC-PERPETUAL",      # Litecoin
    "LINK-PERPETUAL",     # Chainlink
    "DOT-PERPETUAL",      # Polkadot
    "UNI-PERPETUAL",      # Uniswap
    "ATOM-PERPETUAL",     # Cosmos
    "MANA-PERPETUAL",     # Decentraland
    "APE-PERPETUAL",      # ApeCoin
    "FTM-PERPETUAL",      # Fantom
    "SUI-PERPETUAL",      # Sui
    "TRX-PERPETUAL",      # Tron
    "BCH-PERPETUAL"       # Bitcoin Cash
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== DERIBIT API FUNCTIONS ========================
class DeribitFetcher:
    def __init__(self):
        self.session = None
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_instruments(self) -> List[str]:
        """Return hardcoded top coins list"""
        logger.info(f"📋 Using {len(TOP_COINS)} hardcoded perpetual contracts:")
        for i, symbol in enumerate(TOP_COINS, 1):
            logger.info(f"   {i}. {symbol}")
        return TOP_COINS
    
    async def get_candles(self, symbol: str, count: int = CANDLES_FOR_ANALYSIS) -> pd.DataFrame:
        """Fetch OHLCV candles for a symbol"""
        try:
            url = f"{DERIBIT_API_URL}/get_tradingview_chart_data"
            
            # Calculate start time (30min * count candles)
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (30 * 60 * 1000 * count)
            
            params = {
                "instrument_name": symbol,
                "resolution": TIMEFRAME,
                "start_timestamp": start_time,
                "end_timestamp": end_time
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("result"):
                    result = data["result"]
                    
                    df = pd.DataFrame({
                        'timestamp': result['ticks'],
                        'open': result['open'],
                        'high': result['high'],
                        'low': result['low'],
                        'close': result['close'],
                        'volume': result['volume']
                    })
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    return df
                
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

# ======================== CANDLESTICK PATTERN DETECTION ========================
class CandlestickPatterns:
    @staticmethod
    def is_pin_bar_bullish(candle: pd.Series) -> bool:
        """Bullish Pin Bar / Hammer detection"""
        body = abs(candle['close'] - candle['open'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        
        # Long lower wick (2x body), small upper wick
        return (
            lower_wick > 2 * body and
            upper_wick < body * 0.3 and
            candle['close'] > candle['open']  # Bullish close
        )
    
    @staticmethod
    def is_shooting_star_bearish(candle: pd.Series) -> bool:
        """Bearish Shooting Star / Inverted Hammer"""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return (
            upper_wick > 2 * body and
            lower_wick < body * 0.3 and
            candle['close'] < candle['open']  # Bearish close
        )
    
    @staticmethod
    def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
        """Bullish Engulfing Pattern"""
        prev_bearish = prev['close'] < prev['open']
        curr_bullish = curr['close'] > curr['open']
        
        return (
            prev_bearish and
            curr_bullish and
            curr['close'] > prev['open'] and
            curr['open'] < prev['close']
        )
    
    @staticmethod
    def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
        """Bearish Engulfing Pattern"""
        prev_bullish = prev['close'] > prev['open']
        curr_bearish = curr['close'] < curr['open']
        
        return (
            prev_bullish and
            curr_bearish and
            curr['close'] < prev['open'] and
            curr['open'] > prev['close']
        )
    
    @staticmethod
    def is_inside_bar(prev: pd.Series, curr: pd.Series) -> bool:
        """Inside Bar (Compression)"""
        return (
            curr['high'] < prev['high'] and
            curr['low'] > prev['low']
        )

# ======================== SUPPORT/RESISTANCE DETECTION ========================
class SRDetector:
    def __init__(self, zone_percent: float = SR_ZONE_PERCENT):
        self.zone_percent = zone_percent
    
    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """Find swing highs and lows"""
        supports = []
        resistances = []
        
        for i in range(window, len(df) - window):
            # Swing High (Resistance)
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                resistances.append(df['high'].iloc[i])
            
            # Swing Low (Support)
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                supports.append(df['low'].iloc[i])
        
        return {
            'supports': supports,
            'resistances': resistances
        }
    
    def cluster_levels(self, levels: List[float], current_price: float) -> List[float]:
        """Cluster similar price levels into zones"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # If within zone_percent, add to cluster
            if (level - current_cluster[-1]) / current_cluster[-1] <= self.zone_percent:
                current_cluster.append(level)
            else:
                # Store cluster average
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Add last cluster
        clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def count_touches(self, df: pd.DataFrame, level: float) -> int:
        """Count how many times price touched a level"""
        zone_high = level * (1 + self.zone_percent)
        zone_low = level * (1 - self.zone_percent)
        
        touches = 0
        for _, candle in df.iterrows():
            if zone_low <= candle['low'] <= zone_high or zone_low <= candle['high'] <= zone_high:
                touches += 1
        
        return touches
    
    def is_ath_atl(self, df: pd.DataFrame, level: float, level_type: str) -> bool:
        """Check if level is All Time High/Low"""
        if level_type == "resistance":
            return level >= df['high'].max() * 0.999  # 0.1% tolerance
        else:
            return level <= df['low'].min() * 1.001

    def detect_sr_levels(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Main S/R detection with validation"""
        current_price = df['close'].iloc[-1]
        
        # Find swing points
        swing_points = self.find_swing_points(df)
        
        # Cluster levels
        support_levels = self.cluster_levels(swing_points['supports'], current_price)
        resistance_levels = self.cluster_levels(swing_points['resistances'], current_price)
        
        # Validate levels
        valid_supports = []
        valid_resistances = []
        
        # Check supports
        for level in support_levels:
            touches = self.count_touches(df, level)
            is_ath_atl = self.is_ath_atl(df, level, "support")
            
            # ATH/ATL = instant valid, OR 2+ touches, OR 1 touch with pattern
            if is_ath_atl or touches >= MIN_TOUCHES_NORMAL:
                valid_supports.append({
                    'level': level,
                    'touches': touches,
                    'is_ath_atl': is_ath_atl,
                    'distance_percent': ((current_price - level) / level) * 100
                })
        
        # Check resistances
        for level in resistance_levels:
            touches = self.count_touches(df, level)
            is_ath_atl = self.is_ath_atl(df, level, "resistance")
            
            if is_ath_atl or touches >= MIN_TOUCHES_NORMAL:
                valid_resistances.append({
                    'level': level,
                    'touches': touches,
                    'is_ath_atl': is_ath_atl,
                    'distance_percent': ((level - current_price) / current_price) * 100
                })
        
        return {
            'supports': valid_supports,
            'resistances': valid_resistances
        }

# ======================== TRADING SIGNAL GENERATOR ========================
class SignalGenerator:
    def __init__(self):
        self.patterns = CandlestickPatterns()
        self.sr_detector = SRDetector()
    
    def check_volume_spike(self, df: pd.DataFrame) -> bool:
        """Check if recent volume is significantly higher"""
        recent_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-50:-1].mean()
        
        return recent_volume > avg_volume * MIN_VOLUME_SPIKE
    
    def analyze_symbol(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Complete analysis for a symbol"""
        if len(df) < 50:
            return None
        
        # Detect S/R levels
        sr_levels = self.sr_detector.detect_sr_levels(df)
        
        # Get last 2 candles
        curr_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        current_price = curr_candle['close']
        
        # Check volume spike
        volume_spike = self.check_volume_spike(df)
        
        # Check for signals near S/R levels
        signal = None
        
        # Check Support levels
        for support in sr_levels['supports']:
            distance = abs(support['distance_percent'])
            
            # Price near support (within 1%)
            if distance <= 1.0 and current_price < support['level']:
                # Check for bullish patterns
                if self.patterns.is_pin_bar_bullish(curr_candle):
                    signal = {
                        'symbol': symbol,
                        'type': 'BUY',
                        'pattern': 'Bullish Pin Bar',
                        'sr_level': support['level'],
                        'sr_type': 'Support',
                        'price': current_price,
                        'distance_percent': distance,
                        'touches': support['touches'],
                        'is_ath_atl': support['is_ath_atl'],
                        'volume_spike': volume_spike,
                        'timestamp': df.index[-1]
                    }
                    break
                
                elif self.patterns.is_bullish_engulfing(prev_candle, curr_candle):
                    signal = {
                        'symbol': symbol,
                        'type': 'BUY',
                        'pattern': 'Bullish Engulfing',
                        'sr_level': support['level'],
                        'sr_type': 'Support',
                        'price': current_price,
                        'distance_percent': distance,
                        'touches': support['touches'],
                        'is_ath_atl': support['is_ath_atl'],
                        'volume_spike': volume_spike,
                        'timestamp': df.index[-1]
                    }
                    break
        
        # Check Resistance levels
        if not signal:
            for resistance in sr_levels['resistances']:
                distance = abs(resistance['distance_percent'])
                
                # Price near resistance (within 1%)
                if distance <= 1.0 and current_price > resistance['level']:
                    # Check for bearish patterns
                    if self.patterns.is_shooting_star_bearish(curr_candle):
                        signal = {
                            'symbol': symbol,
                            'type': 'SELL',
                            'pattern': 'Shooting Star',
                            'sr_level': resistance['level'],
                            'sr_type': 'Resistance',
                            'price': current_price,
                            'distance_percent': distance,
                            'touches': resistance['touches'],
                            'is_ath_atl': resistance['is_ath_atl'],
                            'volume_spike': volume_spike,
                            'timestamp': df.index[-1]
                        }
                        break
                    
                    elif self.patterns.is_bearish_engulfing(prev_candle, curr_candle):
                        signal = {
                            'symbol': symbol,
                            'type': 'SELL',
                            'pattern': 'Bearish Engulfing',
                            'sr_level': resistance['level'],
                            'sr_type': 'Resistance',
                            'price': current_price,
                            'distance_percent': distance,
                            'touches': resistance['touches'],
                            'is_ath_atl': resistance['is_ath_atl'],
                            'volume_spike': volume_spike,
                            'timestamp': df.index[-1]
                        }
                        break
        
        return signal

# ======================== CHART GENERATION ========================
class ChartGenerator:
    @staticmethod
    def create_chart(df: pd.DataFrame, signal: Dict, sr_levels: Dict) -> str:
        """Create TradingView-style chart with white background"""
        # Take last CANDLES_FOR_CHART candles
        chart_df = df.tail(CANDLES_FOR_CHART).copy()
        
        # Prepare S/R lines
        support_lines = [s['level'] for s in sr_levels['supports']]
        resistance_lines = [r['level'] for r in sr_levels['resistances']]
        
        # Create horizontal lines dict
        hlines = {}
        for i, level in enumerate(support_lines[:3]):  # Top 3 supports
            hlines[level] = {'color': 'green', 'linewidth': 1.5, 'linestyle': '--'}
        
        for i, level in enumerate(resistance_lines[:3]):  # Top 3 resistances
            hlines[level] = {'color': 'red', 'linewidth': 1.5, 'linestyle': '--'}
        
        # Add signal level
        hlines[signal['sr_level']] = {'color': 'blue', 'linewidth': 2, 'linestyle': '-'}
        
        # Create style with white background
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in',
            alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            gridcolor='#e0e0e0',
            facecolor='white',
            figcolor='white',
            y_on_right=False
        )
        
        # Save chart
        filename = f"/home/claude/chart_{signal['symbol'].replace('/', '_')}_{int(datetime.now().timestamp())}.png"
        
        mpf.plot(
            chart_df,
            type='candle',
            style=s,
            volume=True,
            hlines=hlines,
            title=f"{signal['symbol']} - {signal['pattern']} @ {signal['sr_type']}",
            ylabel='Price',
            ylabel_lower='Volume',
            savefig=filename,
            figsize=(12, 8),
            tight_layout=True
        )
        
        logger.info(f"📊 Chart saved: {filename}")
        return filename

# ======================== TELEGRAM ALERTS ========================
class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_alert(self, signal: Dict, chart_path: str):
        """Send alert with chart to Telegram"""
        try:
            # Prepare message
            msg = f"""
🚨 **{signal['type']} SIGNAL** 🚨

📊 **Symbol:** {signal['symbol']}
📈 **Pattern:** {signal['pattern']}
💰 **Price:** ${signal['price']:.2f}
🎯 **{signal['sr_type']}:** ${signal['sr_level']:.2f}
📍 **Distance:** {signal['distance_percent']:.2f}%
🔢 **Touches:** {signal['touches']} {'(ATH/ATL)' if signal['is_ath_atl'] else ''}
📊 **Volume Spike:** {'✅ YES' if signal['volume_spike'] else '❌ NO'}
⏰ **Time:** {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

{'🔥 **Strong Signal - ATH/ATL Level!**' if signal['is_ath_atl'] else ''}
"""
            
            # Send chart image with caption
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=msg,
                    parse_mode='Markdown'
                )
            
            logger.info(f"✅ Alert sent for {signal['symbol']}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")

# ======================== MAIN BOT ========================
class SRTradingBot:
    def __init__(self):
        self.fetcher = DeribitFetcher()
        self.signal_generator = SignalGenerator()
        self.chart_generator = ChartGenerator()
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_alerts = defaultdict(lambda: None)  # Track last alert time per symbol
    
    async def analyze_all_symbols(self):
        """Analyze all top symbols"""
        symbols = await self.fetcher.get_instruments()
        
        if not symbols:
            logger.warning("⚠️ No symbols to analyze!")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 STARTING ANALYSIS - {len(symbols)} SYMBOLS")
        logger.info(f"{'='*60}\n")
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{idx}/{len(symbols)}] 📊 Analyzing {symbol}...")
                
                # Fetch candles
                df = await self.fetcher.get_candles(symbol, CANDLES_FOR_ANALYSIS)
                
                if df.empty:
                    logger.warning(f"   ⚠️ No data available for {symbol}")
                    continue
                
                logger.info(f"   ✅ Fetched {len(df)} candles | Latest: ${df['close'].iloc[-1]:.2f}")
                
                # Analyze for signals
                signal = self.signal_generator.analyze_symbol(df, symbol)
                
                if signal:
                    # Check if we already alerted recently (avoid spam)
                    last_alert_time = self.last_alerts[symbol]
                    current_time = datetime.now()
                    
                    if last_alert_time is None or (current_time - last_alert_time).seconds > 3600:  # 1 hour cooldown
                        logger.info(f"   🎯 SIGNAL FOUND: {signal['type']} - {signal['pattern']}")
                        logger.info(f"   💰 Price: ${signal['price']:.2f} | S/R: ${signal['sr_level']:.2f}")
                        
                        # Get S/R levels for chart
                        sr_levels = self.signal_generator.sr_detector.detect_sr_levels(df)
                        
                        # Generate chart
                        chart_path = self.chart_generator.create_chart(df, signal, sr_levels)
                        
                        # Send alert
                        await self.alerter.send_alert(signal, chart_path)
                        
                        # Update last alert time
                        self.last_alerts[symbol] = current_time
                        logger.info(f"   ✉️ Alert sent successfully!\n")
                    else:
                        logger.info(f"   ⏸️ Signal found but cooldown active (last alert: {last_alert_time.strftime('%H:%M:%S')})")
                else:
                    logger.info(f"   ℹ️ No trading signals detected\n")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"   ❌ Error analyzing {symbol}: {e}\n")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ ANALYSIS COMPLETE")
        logger.info(f"{'='*60}\n")
    
    async def run(self):
        """Main bot loop"""
        await self.fetcher.create_session()
        
        # Startup banner
        logger.info("\n" + "="*60)
        logger.info("🚀 S/R TRADING BOT STARTED!")
        logger.info("="*60)
        logger.info(f"⏱️  Analysis Interval: Every 30 minutes")
        logger.info(f"📊 Timeframe: 30min candles")
        logger.info(f"📈 Candles per analysis: {CANDLES_FOR_ANALYSIS}")
        logger.info(f"🎯 S/R Zone Width: {SR_ZONE_PERCENT*100}%")
        logger.info(f"📊 Chart Candles: {CANDLES_FOR_CHART}")
        logger.info(f"🔊 Volume Spike Threshold: {MIN_VOLUME_SPIKE}x")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                try:
                    await self.analyze_all_symbols()
                    logger.info(f"⏰ Next analysis in 30 minutes... (at {(datetime.now() + timedelta(minutes=30)).strftime('%H:%M:%S')})\n")
                    await asyncio.sleep(30 * 60)  # 30 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    logger.info("⏳ Retrying in 1 minute...")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        
        finally:
            await self.fetcher.close_session()
            logger.info("👋 Session closed. Goodbye!")

# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    bot = SRTradingBot()
    asyncio.run(bot.run())
