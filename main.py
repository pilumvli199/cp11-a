import os
import asyncio
import logging
from datetime import datetime, timedelta
import redis
import json
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
CANDLE_COUNT = 500  # 500 candles for all timeframes

class DeribitClient:
    """Fetch data from Deribit public API"""
    
    @staticmethod
    def get_candles(symbol: str, timeframe: str, count: int = CANDLE_COUNT) -> pd.DataFrame:
        """Fetch OHLCV data - 500 candles"""
        url = f"{DERIBIT_BASE}/get_tradingview_chart_data"
        
        # Calculate time range for 500 candles
        tf_minutes = int(timeframe)
        days_needed = (count * tf_minutes) // (24 * 60) + 7  # Extra buffer
        
        params = {
            'instrument_name': symbol,
            'resolution': timeframe,
            'start_timestamp': int((datetime.now() - timedelta(days=days_needed)).timestamp() * 1000),
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
            logger.error(f"Error fetching candles for {symbol} {timeframe}m: {e}")
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
                swing_highs.append({
                    'price': df['high'].iloc[i], 
                    'index': i,
                    'timestamp': df.index[i]
                })
            
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                swing_lows.append({
                    'price': df['low'].iloc[i], 
                    'index': i,
                    'timestamp': df.index[i]
                })
        
        return swing_highs[-7:], swing_lows[-7:]  # Last 7 swing points
    
    @staticmethod
    def find_trendlines(swing_points: List, df: pd.DataFrame, is_resistance: bool = True) -> List[Dict]:
        """Find trendlines from swing points"""
        if len(swing_points) < 2:
            return []
        
        trendlines = []
        
        # Try connecting recent swing points
        for i in range(len(swing_points) - 1):
            for j in range(i + 1, len(swing_points)):
                point1 = swing_points[i]
                point2 = swing_points[j]
                
                # Calculate slope
                x_diff = point2['index'] - point1['index']
                y_diff = point2['price'] - point1['price']
                
                if x_diff == 0:
                    continue
                
                slope = y_diff / x_diff
                
                # Project to current
                current_index = len(df) - 1
                projected_price = point2['price'] + slope * (current_index - point2['index'])
                
                trendlines.append({
                    'start_price': point1['price'],
                    'end_price': projected_price,
                    'start_index': point1['index'],
                    'end_index': current_index,
                    'slope': slope,
                    'type': 'resistance' if is_resistance else 'support'
                })
        
        # Return most relevant trendlines
        return sorted(trendlines, key=lambda x: abs(x['slope']))[:3]
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick and chart patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        candle_range = last['high'] - last['low']
        
        # Candlestick Patterns
        if candle_range > 0:
            # Doji
            if body < candle_range * 0.1:
                patterns.append({'type': 'candlestick', 'name': 'Doji', 'signal': 'neutral'})
            
            # Hammer
            if lower_wick > body * 2 and upper_wick < body * 0.5 and last['close'] > last['open']:
                patterns.append({'type': 'candlestick', 'name': 'Hammer', 'signal': 'bullish'})
            
            # Shooting Star
            if upper_wick > body * 2 and lower_wick < body * 0.5 and last['close'] < last['open']:
                patterns.append({'type': 'candlestick', 'name': 'Shooting Star', 'signal': 'bearish'})
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and 
                last['close'] > last['open'] and 
                last['close'] > prev['open'] and 
                last['open'] < prev['close']):
                patterns.append({'type': 'candlestick', 'name': 'Bullish Engulfing', 'signal': 'bullish'})
            
            # Bearish Engulfing
            if (prev['close'] > prev['open'] and 
                last['close'] < last['open'] and 
                last['close'] < prev['open'] and 
                last['open'] > prev['close']):
                patterns.append({'type': 'candlestick', 'name': 'Bearish Engulfing', 'signal': 'bearish'})
        
        # Chart Patterns (simplified)
        if len(df) >= 20:
            recent = df.tail(20)
            
            # Triangle detection (price consolidation)
            highs = recent['high'].values
            lows = recent['low'].values
            
            if max(highs[-10:]) < max(highs[:10]) and min(lows[-10:]) > min(lows[:10]):
                patterns.append({'type': 'chart', 'name': 'Symmetrical Triangle', 'signal': 'breakout_pending'})
            
            # Ascending Triangle
            elif abs(max(highs[-10:]) - max(highs[:10])) < (max(highs) * 0.01) and min(lows[-10:]) > min(lows[:10]):
                patterns.append({'type': 'chart', 'name': 'Ascending Triangle', 'signal': 'bullish'})
            
            # Descending Triangle
            elif abs(min(lows[-10:]) - min(lows[:10])) < (min(lows) * 0.01) and max(highs[-10:]) < max(highs[:10]):
                patterns.append({'type': 'chart', 'name': 'Descending Triangle', 'signal': 'bearish'})
        
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
        
        if volume_by_price:
            poc_bin = max(volume_by_price, key=volume_by_price.get)
            poc_price = df['low'].min() + (poc_bin * bin_size)
            
            return {
                'poc': poc_price,
                'high_volume_nodes': sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        return {'poc': None, 'high_volume_nodes': []}

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
        try:
            for key in redis_client.scan_iter(match=pattern):
                timestamp = int(key.split(':')[-1])
                if timestamp >= cutoff:
                    data = json.loads(redis_client.get(key))
                    data['timestamp'] = timestamp
                    history.append(data)
        except Exception as e:
            logger.error(f"Redis error: {e}")
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    @staticmethod
    def analyze_oi_trend(symbol: str) -> Dict:
        """Analyze OI trend over last 2 hours"""
        history = OITracker.get_oi_history(symbol, hours=2)
        
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'change': 0, 'supporting_sr': None}
        
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
            'previous_oi': old_oi,
            'supporting_sr': None  # Will be set by analyzer
        }

class ChartGenerator:
    """Generate annotated charts"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, analysis: Dict, symbol: str) -> io.BytesIO:
        """Create chart with S/R, trendlines, patterns marked"""
        
        # Setup white background style
        mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in',
            alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#e0e0e0',
            facecolor='white',
            figcolor='white',
            y_on_right=False
        )
        
        # Create figure
        fig, axes = mpf.plot(
            df.tail(100),  # Show last 100 candles
            type='candle',
            style=s,
            volume=True,
            returnfig=True,
            figsize=(14, 8),
            title=f"\n{symbol} - {analysis.get('trade_type', 'Analysis')}",
            ylabel='Price ($)',
            ylabel_lower='Volume'
        )
        
        ax = axes[0]
        
        # Get data range for plotting
        plot_df = df.tail(100)
        x_range = range(len(plot_df))
        
        # Plot Support levels (green)
        if 'swing_lows_30m' in analysis:
            for swing in analysis['swing_lows_30m'][-3:]:  # Last 3 swing lows
                price = swing['price']
                ax.axhline(y=price, color='#4caf50', linestyle='--', linewidth=1.5, alpha=0.7, label='Support 30m')
        
        # Plot Resistance levels (red)
        if 'swing_highs_30m' in analysis:
            for swing in analysis['swing_highs_30m'][-3:]:
                price = swing['price']
                ax.axhline(y=price, color='#f44336', linestyle='--', linewidth=1.5, alpha=0.7, label='Resistance 30m')
        
        # Plot 4H S/R (thicker lines)
        if analysis.get('support_4h'):
            ax.axhline(y=analysis['support_4h'], color='#2e7d32', linestyle='-', linewidth=2, label='Support 4H')
        
        if analysis.get('resistance_4h'):
            ax.axhline(y=analysis['resistance_4h'], color='#c62828', linestyle='-', linewidth=2, label='Resistance 4H')
        
        # Mark OI supported S/R (blue)
        if analysis.get('oi_trend', {}).get('supporting_sr'):
            sr_price = analysis['oi_trend']['supporting_sr']
            ax.axhline(y=sr_price, color='#2196f3', linestyle='-.', linewidth=2.5, label='OI Support/Resistance')
        
        # Mark current price
        current_price = df['close'].iloc[-1]
        ax.axhline(y=current_price, color='#ff9800', linestyle=':', linewidth=2, label=f'Current: ${current_price:.2f}')
        
        # Add pattern annotations
        if analysis.get('patterns'):
            pattern_text = "\n".join([p['name'] for p in analysis['patterns'][:3]])
            ax.text(
                0.02, 0.98, 
                f"Patterns:\n{pattern_text}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10
            )
        
        # Add trade signal box
        if analysis.get('trade_signal'):
            signal = analysis['trade_signal']
            signal_color = '#4caf50' if 'LONG' in signal else '#f44336' if 'SHORT' in signal else '#9e9e9e'
            
            signal_text = f"SIGNAL: {signal}\n"
            if analysis.get('entry_price'):
                signal_text += f"Entry: ${analysis['entry_price']:.2f}\n"
            if analysis.get('sl_price'):
                signal_text += f"SL: ${analysis['sl_price']:.2f}\n"
            if analysis.get('target_price'):
                signal_text += f"Target: ${analysis['target_price']:.2f}"
            
            ax.text(
                0.98, 0.98,
                signal_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.3, edgecolor=signal_color, linewidth=2),
                fontsize=11,
                fontweight='bold'
            )
        
        # Clean up legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        
        # Save to BytesIO
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        return buf

class TradeAnalyzer:
    """Main trade analysis engine"""
    
    @staticmethod
    def analyze_setup(symbol: str) -> Dict:
        """Comprehensive trade analysis with 500 candles per TF"""
        logger.info(f"Analyzing {symbol}...")
        
        # Fetch multi-timeframe data (500 candles each)
        df_30m = DeribitClient.get_candles(symbol, '30', CANDLE_COUNT)
        df_1h = DeribitClient.get_candles(symbol, '60', CANDLE_COUNT)
        df_4h = DeribitClient.get_candles(symbol, '240', CANDLE_COUNT)
        
        if df_30m.empty or df_1h.empty or df_4h.empty:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        logger.info(f"{symbol}: Loaded {len(df_30m)} x 30m, {len(df_1h)} x 1h, {len(df_4h)} x 4h candles")
        
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
        
        # Determine OI supported S/R
        oi_sr = None
        if oi_trend['trend'] in ['increasing', 'strongly_increasing']:
            # OI increasing - check if near resistance (bearish) or support (bullish)
            recent_high = df_30m['high'].tail(10).max()
            recent_low = df_30m['low'].tail(10).min()
            
            if abs(current_price - recent_high) < abs(current_price - recent_low):
                oi_sr = recent_high  # Near resistance
            else:
                oi_sr = recent_low  # Near support
        
        oi_trend['supporting_sr'] = oi_sr
        
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
            'df_30m': df_30m,  # For chart generation
            'valid': False
        }
        
        return analysis
    
    @staticmethod
    def get_ai_analysis(analysis: Dict) -> Dict:
        """Get GPT-4o mini analysis with structured output"""
        
        patterns_text = "\n".join([f"- {p['name']} ({p['signal']})" for p in analysis['patterns']]) if analysis['patterns'] else "None"
        
        prompt = f"""Analyze this crypto trading setup for breakout/breakdown:

Symbol: {analysis['symbol']}
Price: ${analysis['current_price']:.2f}
Timeframe: 30min

PATTERNS DETECTED:
{patterns_text}

VOLUME: {analysis['volume_ratio']}x average (need >1.5x)

OI ANALYSIS:
- Trend: {analysis['oi_trend']['trend']}
- Change: {analysis['oi_trend']['change']}%
- Supporting S/R: ${analysis['oi_trend']['supporting_sr']:.2f if analysis['oi_trend']['supporting_sr'] else 'N/A'}

SUPPORT/RESISTANCE:
- 30m swing low: ${analysis['swing_lows_30m'][-1]['price']:.2f if analysis['swing_lows_30m'] else 'N/A'}
- 30m swing high: ${analysis['swing_highs_30m'][-1]['price']:.2f if analysis['swing_highs_30m'] else 'N/A'}
- 4h support: ${analysis['support_4h']:.2f if analysis['support_4h'] else 'N/A'}
- 4h resistance: ${analysis['resistance_4h']:.2f if analysis['resistance_4h'] else 'N/A'}

RULES (STRICT):
1. Body close beyond S/R required
2. Wicks < 30% of candle
3. Volume MUST be >1.5x average
4. OI should support direction
5. Multi-TF alignment needed

TASK: Determine if this is valid LONG/SHORT/NO_TRADE.

Response format (strict):
SIGNAL: [LONG or SHORT or NO_TRADE]
ENTRY: [price]
SL: [recent swing level]
TARGET: [price with min 1:2 R:R]
PATTERN: [which pattern triggered trade]
REASON: [1-2 sentences only]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert crypto trader. Respond in exact format requested. Be strict with rules."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.2
            )
            
            ai_text = response.choices[0].message.content
            
            # Parse AI response
            result = {
                'signal': 'NO_TRADE',
                'entry': None,
                'sl': None,
                'target': None,
                'pattern': 'None',
                'reason': ai_text
            }
            
            for line in ai_text.split('\n'):
                if 'SIGNAL:' in line:
                    result['signal'] = line.split('SIGNAL:')[-1].strip()
                elif 'ENTRY:' in line:
                    try:
                        result['entry'] = float(line.split('$')[-1].strip())
                    except:
                        pass
                elif 'SL:' in line:
                    try:
                        result['sl'] = float(line.split('$')[-1].strip())
                    except:
                        pass
                elif 'TARGET:' in line:
                    try:
                        result['target'] = float(line.split('$')[-1].strip())
                    except:
                        pass
                elif 'PATTERN:' in line:
                    result['pattern'] = line.split('PATTERN:')[-1].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"GPT API error: {e}")
            return {
                'signal': 'ERROR',
                'reason': f'AI analysis failed: {str(e)}'
            }

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
            logger.info("Trade counter reset for new day")
    
    async def scan_markets(self, context: ContextTypes.DEFAULT_TYPE):
        """Scan all symbols for setups"""
        self.reset_daily_counter()
        
        if self.trade_count_today >= MAX_TRADES_PER_DAY:
            logger.info(f"Daily limit reached: {self.trade_count_today}/{MAX_TRADES_PER_DAY}")
            return
        
        for symbol in SYMBOLS:
            try:
                logger.info(f"Scanning {symbol}...")
                analysis = TradeAnalyzer.analyze_setup(symbol)
                
                if not analysis.get('valid', True):  # Skip if data issues
                    continue
                
                # Basic filters
                if (analysis.get('volume_ratio', 0) < 1.5 or 
                    not analysis.get('patterns')):
                    logger.info(f"{symbol}: No patterns or low volume")
                    continue
                
                # Get AI analysis
                ai_result = TradeAnalyzer.get_ai_analysis(analysis)
                
                # Check if valid signal
                if ai_result['signal'] in ['LONG', 'SHORT']:
                    self.trade_count_today += 1
                    
                    # Add AI results to analysis for chart
                    analysis['trade_signal'] = ai_result['signal']
                    analysis['entry_price'] = ai_result['entry']
                    analysis['sl_price'] = ai_result['sl']
                    analysis['target_price'] = ai_result['target']
                    analysis['trade_type'] = ai_result['pattern']
                    
                    await self.send_alert(context, symbol, analysis, ai_result)
                else:
                    logger.info(f"{symbol}: {ai_result['signal']} - {ai_result.get('reason', '')[:50]}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
            
            await asyncio.sleep(3)  # Rate limit between symbols
    
    async def send_alert(self, context: ContextTypes.DEFAULT_TYPE, symbol: str, analysis: Dict, ai_result: Dict):
        """Send trade alert with chart to Telegram"""
        
        # Generate chart
        try:
            chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, symbol)
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            chart_buf = None
        
        # Prepare text message
        patterns_text = ", ".join([p['name'] for p in analysis['patterns'][:3]])
        
        oi_emoji = "📈" if "increasing" in analysis['oi_trend']['trend'] else "📉" if "decreasing" in analysis['oi_trend']['trend'] else "➡️"
        signal_emoji = "🟢" if ai_result['signal'] == 'LONG' else "🔴"
        
        message = f"""{signal_emoji} **{analysis['symbol']} - {ai_result['signal']} SETUP**

📊 **Pattern:** {ai_result['pattern']}
💰 **Price:** ${analysis['current_price']:.2f}

🎯 **TRADE DETAILS:**
├ Entry: ${ai_result['entry']:.2f}
├ Stop Loss: ${ai_result['sl']:.2f}
├ Target: ${ai_result['target']:.2f}
└ R:R = 1:{abs((ai_result['target'] - ai_result['entry']) / (ai_result['entry'] - ai_result['sl'])):.1f}

✅ **CONFIRMATIONS:**
├ Patterns: {patterns_text}
├ Volume: {analysis['volume_ratio']}x avg
└ OI: {oi_emoji} {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']:+.1f}%)

📈 **SUPPORT & RESISTANCE:**
├ 30m Support: ${analysis['swing_lows_30m'][-1]['price']:.2f if analysis['swing_lows_30m'] else 'N/A'}
├ 30m Resistance: ${analysis['swing_highs_30m'][-1]['price']:.2f if analysis['swing_highs_30m'] else 'N/A'}
├ 4H Support: ${analysis['support_4h']:.2f if analysis['support_4h'] else 'N/A'}
└ 4H Resistance: ${analysis['resistance_4h']:.2f if analysis['resistance_4h'] else 'N/A'}

💡 **Analysis:** {ai_result['reason']}

⚠️ Trade #{self.trade_count_today}/{MAX_TRADES_PER_DAY} today
"""
        
        try:
            # Send chart first
            if chart_buf:
                await context.bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=chart_buf,
                    caption=message,
                    parse_mode='Markdown'
                )
            else:
                # If chart fails, send text only
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
            
            logger.info(f"Alert sent for {symbol}: {ai_result['signal']}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

# Bot commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    await update.message.reply_text(
        "🤖 **Trading Bot Active!**\n\n"
        "📊 **Tracking:** BTC & ETH (Deribit)\n"
        "⏱ **Timeframes:** 30m, 1hr, 4hr (500 candles each)\n"
        "🎯 **Max Trades:** 8 per day\n"
        "📈 **Scan Interval:** Every 30 minutes\n\n"
        "**Commands:**\n"
        "/status - Check bot status\n"
        "/scan - Manual scan now\n"
        "/analyze BTC - Analyze specific symbol\n\n"
        "🚀 Bot will automatically scan and alert on valid setups!",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status command"""
    bot = context.bot_data.get('trading_bot')
    if bot:
        await update.message.reply_text(
            f"📊 **Bot Status:**\n\n"
            f"✅ Active and Running\n"
            f"📈 Trades Today: {bot.trade_count_today}/{MAX_TRADES_PER_DAY}\n"
            f"⏱ Scan Interval: 30 minutes\n"
            f"💾 Using Redis for OI tracking\n"
            f"📊 Candles per TF: {CANDLE_COUNT}\n\n"
            f"Next scan in ~30 mins",
            parse_mode='Markdown'
        )

async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual scan command"""
    await update.message.reply_text("🔍 Starting manual scan...")
    bot = context.bot_data.get('trading_bot')
    if bot:
        await bot.scan_markets(context)
        await update.message.reply_text("✅ Scan complete!")

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze specific symbol"""
    if not context.args:
        await update.message.reply_text("Usage: /analyze BTC or /analyze ETH")
        return
    
    symbol_input = context.args[0].upper()
    symbol = f"{symbol_input}-PERPETUAL"
    
    if symbol not in SYMBOLS:
        await update.message.reply_text(f"❌ Invalid symbol. Use: BTC or ETH")
        return
    
    await update.message.reply_text(f"🔍 Analyzing {symbol}...")
    
    try:
        analysis = TradeAnalyzer.analyze_setup(symbol)
        
        if not analysis.get('valid', True):
            await update.message.reply_text(f"❌ Cannot analyze {symbol}: {analysis.get('reason', 'Unknown error')}")
            return
        
        ai_result = TradeAnalyzer.get_ai_analysis(analysis)
        
        # Add AI results for chart
        analysis['trade_signal'] = ai_result['signal']
        analysis['entry_price'] = ai_result.get('entry')
        analysis['sl_price'] = ai_result.get('sl')
        analysis['target_price'] = ai_result.get('target')
        analysis['trade_type'] = ai_result.get('pattern', 'Analysis')
        
        # Generate chart
        chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, symbol)
        
        # Prepare message
        patterns_text = ", ".join([p['name'] for p in analysis['patterns']]) if analysis['patterns'] else "None"
        
        message = f"""📊 **{symbol} Analysis**

💰 Price: ${analysis['current_price']:.2f}
📈 Patterns: {patterns_text}
📊 Volume: {analysis['volume_ratio']}x avg
🔄 OI: {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']:+.1f}%)

🤖 **AI Signal:** {ai_result['signal']}
💡 {ai_result.get('reason', 'No reason provided')}
"""
        
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=chart_buf,
            caption=message,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Error in analyze command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")

def main():
    """Main function"""
    logger.info("Initializing Trading Bot...")
    
    # Validate environment variables
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not CHAT_ID:
        logger.error("Missing required environment variables!")
        logger.error("Required: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, TELEGRAM_CHAT_ID")
        return
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.warning("Bot will continue but OI tracking may not work properly")
    
    # Initialize bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    trading_bot = TradingBot()
    application.bot_data['trading_bot'] = trading_bot
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("scan", scan_now))
    application.add_handler(CommandHandler("analyze", analyze_symbol))
    
    # Schedule market scans every 30 mins
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(
            trading_bot.scan_markets,
            interval=1800,  # 30 minutes
            first=10  # Start after 10 seconds
        )
        logger.info("✅ Job queue configured - scanning every 30 mins")
    else:
        logger.error("❌ Job queue not available!")
    
    # Start bot
    logger.info("🚀 Bot starting...")
    logger.info(f"📊 Tracking: {', '.join(SYMBOLS)}")
    logger.info(f"⏱ Scan interval: 30 minutes")
    logger.info(f"📈 Candles per TF: {CANDLE_COUNT}")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
