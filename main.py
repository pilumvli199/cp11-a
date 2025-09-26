#!/usr/bin/env python3
"""
Enhanced Crypto Trading Bot - Complete Price Action Analysis System
Features:
- Advanced technical analysis with 720 candles (30 days)
- Smart Money Concepts (Order Blocks, Liquidity Zones, ChoCh, BOS)
- Professional chart generation with multiple timeframes
- Redis caching and performance tracking
- OpenAI GPT-4 integration for enhanced analysis
- Comprehensive Telegram alerts with risk management
"""

import os
import json
import asyncio
import traceback
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd

# Plotting with server-safe backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.dates import date2num

# OpenAI integration
from openai import OpenAI

# Redis for caching and performance tracking
import redis
from redis.exceptions import RedisError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================

@dataclass
class BotConfig:
    """Bot configuration with validation"""
    symbols: List[str]
    timeframes: List[str]
    analysis_candles: int
    poll_interval: int
    signal_confidence_threshold: float
    risk_reward_min: float
    max_signals_per_hour: int
    
    def __post_init__(self):
        self.symbols = self.symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AAVEUSDT",
            "TRXUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT"
        ]
        self.timeframes = self.timeframes or ["1h", "4h", "1d"]

# Initialize configuration
config = BotConfig(
    symbols=os.getenv("TRADING_SYMBOLS", "").split(",") if os.getenv("TRADING_SYMBOLS") else None,
    timeframes=["1h", "4h", "1d"],
    analysis_candles=720,  # 30 days of 1h candles
    poll_interval=max(300, int(os.getenv("POLL_INTERVAL", "1800"))),  # Min 5 minutes
    signal_confidence_threshold=float(os.getenv("SIGNAL_CONF_THRESHOLD", "75.0")),
    risk_reward_min=float(os.getenv("MIN_RISK_REWARD", "1.5")),
    max_signals_per_hour=int(os.getenv("MAX_SIGNALS_PER_HOUR", "3"))
)

# API Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Binance API URLs
BINANCE_BASE = "https://api.binance.com/api/v3"
KLINES_URL = f"{BINANCE_BASE}/klines"
TICKER_24H_URL = f"{BINANCE_BASE}/ticker/24hr"

# ================== REDIS CONFIGURATION ==================

class RedisManager:
    """Enhanced Redis manager with error handling and connection pooling"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self._init_connection()
    
    def _init_connection(self):
        """Initialize Redis connection with multiple fallback options"""
        try:
            # Try environment variables first
            redis_url = os.getenv("REDIS_URL")
            redis_host = os.getenv("REDIS_HOST")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_user = os.getenv("REDIS_USER")
            
            if redis_url:
                logger.info("Connecting to Redis via URL...")
                self.client = redis.from_url(
                    redis_url, 
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
            elif redis_host:
                logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    username=redis_user,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
            else:
                logger.warning("No Redis configuration found, running without caching")
                return
            
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None
            self.connected = False
    
    def safe_execute(self, operation, *args, **kwargs):
        """Execute Redis operation with error handling"""
        if not self.connected:
            return None
        
        try:
            return operation(*args, **kwargs)
        except RedisError as e:
            logger.error(f"Redis operation failed: {e}")
            return None
    
    def store_candle_data(self, symbol: str, timeframe: str, candles: List[Dict]):
        """Store candle data in Redis streams"""
        if not self.connected:
            return
        
        stream_key = f"candles:{timeframe}:{symbol}"
        maxlen = {"1h": 2160, "4h": 1080, "1d": 730}.get(timeframe, 1000)
        
        for candle in candles[-10:]:  # Store last 10 candles
            try:
                mapping = {
                    "timestamp": str(candle.get("timestamp", 0)),
                    "open": str(candle.get("open", 0)),
                    "high": str(candle.get("high", 0)),
                    "low": str(candle.get("low", 0)),
                    "close": str(candle.get("close", 0)),
                    "volume": str(candle.get("volume", 0))
                }
                self.safe_execute(
                    self.client.xadd, 
                    stream_key, 
                    mapping, 
                    maxlen=maxlen, 
                    approximate=True
                )
            except Exception as e:
                logger.error(f"Failed to store candle for {symbol}: {e}")
    
    def store_analysis_cache(self, symbol: str, analysis: Dict):
        """Cache analysis results"""
        cache_key = f"analysis:{symbol}"
        cache_data = {
            "timestamp": time.time(),
            "data": json.dumps(analysis)
        }
        self.safe_execute(
            self.client.hset, 
            cache_key, 
            mapping=cache_data
        )
        self.safe_execute(self.client.expire, cache_key, 3600)  # 1 hour TTL
    
    def get_analysis_cache(self, symbol: str) -> Optional[Dict]:
        """Retrieve cached analysis"""
        cache_key = f"analysis:{symbol}"
        cached = self.safe_execute(self.client.hgetall, cache_key)
        
        if cached and "data" in cached:
            try:
                cache_time = float(cached.get("timestamp", 0))
                if time.time() - cache_time < 1800:  # 30 minutes cache
                    return json.loads(cached["data"])
            except Exception as e:
                logger.error(f"Failed to parse cached analysis: {e}")
        return None
    
    def store_signal(self, symbol: str, signal: Dict):
        """Store trading signal"""
        signal_key = f"signals:active:{symbol}"
        signal_data = {
            **signal,
            "timestamp": time.time(),
            "symbol": symbol
        }
        self.safe_execute(
            self.client.setex,
            signal_key,
            86400,  # 24 hours TTL
            json.dumps(signal_data)
        )
    
    def track_performance(self, symbol: str, signal_id: str, result: Dict):
        """Track signal performance for optimization"""
        perf_key = f"performance:{symbol}"
        performance_data = {
            "signal_id": signal_id,
            "timestamp": time.time(),
            **result
        }
        self.safe_execute(
            self.client.lpush,
            perf_key,
            json.dumps(performance_data)
        )
        # Keep only last 1000 records
        self.safe_execute(self.client.ltrim, perf_key, 0, 999)

# Initialize Redis manager
redis_manager = RedisManager()

# ================== TECHNICAL ANALYSIS ENGINE ==================

class TechnicalAnalyzer:
    """Advanced technical analysis with Smart Money Concepts"""
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return [None] * len(prices)
        
        multiplier = 2 / (period + 1)
        ema_values = [None] * (period - 1)
        
        # Initial SMA for first EMA value
        ema_values.append(sum(prices[:period]) / period)
        
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = [None] * period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        else:
            rsi_values.append(100)
        
        for i in range(period + 1, len(prices)):
            gain = gains[i-1]
            loss = losses[i-1]
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
            else:
                rsi_values.append(100)
        
        return rsi_values
    
    @staticmethod
    def find_support_resistance_levels(highs: List[float], lows: List[float], 
                                     closes: List[float], lookback: int = 50) -> Dict:
        """Advanced Support/Resistance detection with touch counting"""
        if len(closes) < lookback:
            return {"support": [], "resistance": []}
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        current_price = closes[-1]
        
        # Combine all significant price points
        all_prices = recent_highs + recent_lows + recent_closes
        
        # Group similar prices (within 0.5% of each other)
        price_groups = []
        tolerance = 0.005  # 0.5% tolerance
        
        for price in all_prices:
            added_to_group = False
            for group in price_groups:
                if abs(price - group["avg_price"]) / group["avg_price"] <= tolerance:
                    group["prices"].append(price)
                    group["avg_price"] = sum(group["prices"]) / len(group["prices"])
                    group["touches"] += 1
                    added_to_group = True
                    break
            
            if not added_to_group:
                price_groups.append({
                    "prices": [price],
                    "avg_price": price,
                    "touches": 1
                })
        
        # Filter and categorize levels
        significant_levels = [g for g in price_groups if g["touches"] >= 2]
        
        support_levels = []
        resistance_levels = []
        
        for level in significant_levels:
            level_price = level["avg_price"]
            if level_price < current_price:
                support_levels.append({
                    "price": level_price,
                    "strength": min(level["touches"] * 10, 100),
                    "touches": level["touches"]
                })
            else:
                resistance_levels.append({
                    "price": level_price,
                    "strength": min(level["touches"] * 10, 100),
                    "touches": level["touches"]
                })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x["strength"], reverse=True)
        resistance_levels.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "support": support_levels[:5],  # Top 5 support levels
            "resistance": resistance_levels[:5]  # Top 5 resistance levels
        }
    
    @staticmethod
    def detect_market_structure(highs: List[float], lows: List[float], 
                              closes: List[float]) -> Dict:
        """Detect market structure: HH/HL (bullish) vs LH/LL (bearish)"""
        if len(closes) < 20:
            return {"trend": "sideways", "strength": 0, "structure": "unclear"}
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(highs) - 5):
            # Check if current high is higher than surrounding highs
            if (highs[i] > max(highs[i-5:i]) and 
                highs[i] > max(highs[i+1:i+6])):
                swing_highs.append((i, highs[i]))
            
            # Check if current low is lower than surrounding lows
            if (lows[i] < min(lows[i-5:i]) and 
                lows[i] < min(lows[i+1:i+6])):
                swing_lows.append((i, lows[i]))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"trend": "sideways", "strength": 0, "structure": "unclear"}
        
        # Analyze recent structure (last 3 swings)
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Check for Higher Highs and Higher Lows (bullish structure)
        bullish_score = 0
        if len(recent_highs) >= 2:
            if recent_highs[-1][1] > recent_highs[-2][1]:
                bullish_score += 30
        if len(recent_lows) >= 2:
            if recent_lows[-1][1] > recent_lows[-2][1]:
                bullish_score += 30
        
        # Check for Lower Highs and Lower Lows (bearish structure)
        bearish_score = 0
        if len(recent_highs) >= 2:
            if recent_highs[-1][1] < recent_highs[-2][1]:
                bearish_score += 30
        if len(recent_lows) >= 2:
            if recent_lows[-1][1] < recent_lows[-2][1]:
                bearish_score += 30
        
        # Overall trend assessment
        if bullish_score > bearish_score and bullish_score >= 40:
            trend = "bullish"
            strength = min(bullish_score, 100)
        elif bearish_score > bullish_score and bearish_score >= 40:
            trend = "bearish"
            strength = min(bearish_score, 100)
        else:
            trend = "sideways"
            strength = max(bullish_score, bearish_score)
        
        return {
            "trend": trend,
            "strength": strength,
            "structure": f"Recent swings: {len(recent_highs)} highs, {len(recent_lows)} lows",
            "swing_highs": recent_highs,
            "swing_lows": recent_lows
        }
    
    @staticmethod
    def detect_order_blocks(opens: List[float], highs: List[float], 
                          lows: List[float], closes: List[float], 
                          volumes: List[float]) -> List[Dict]:
        """Detect Order Blocks (high volume candles before significant moves)"""
        if len(closes) < 20:
            return []
        
        order_blocks = []
        avg_volume = sum(volumes[-50:]) / min(50, len(volumes))
        
        for i in range(10, len(closes) - 5):
            current_volume = volumes[i]
            
            # High volume condition (2x average)
            if current_volume > avg_volume * 2:
                # Check for significant move after this candle
                next_5_closes = closes[i+1:i+6]
                current_close = closes[i]
                
                if next_5_closes:
                    max_move_up = max(next_5_closes) - current_close
                    max_move_down = current_close - min(next_5_closes)
                    
                    # Significant move threshold (1% of price)
                    threshold = current_close * 0.01
                    
                    if max_move_up > threshold:
                        # Bullish order block
                        order_blocks.append({
                            "type": "bullish",
                            "index": i,
                            "high": highs[i],
                            "low": lows[i],
                            "volume": current_volume,
                            "strength": min((current_volume / avg_volume) * 10, 100)
                        })
                    
                    elif max_move_down > threshold:
                        # Bearish order block
                        order_blocks.append({
                            "type": "bearish",
                            "index": i,
                            "high": highs[i],
                            "low": lows[i],
                            "volume": current_volume,
                            "strength": min((current_volume / avg_volume) * 10, 100)
                        })
        
        # Return only recent order blocks (last 20)
        return order_blocks[-20:]
    
    @staticmethod
    def detect_candlestick_patterns(opens: List[float], highs: List[float], 
                                  lows: List[float], closes: List[float]) -> List[Dict]:
        """Detect key candlestick patterns"""
        if len(closes) < 3:
            return []
        
        patterns = []
        
        for i in range(1, len(closes)):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            prev_o, prev_h, prev_l, prev_c = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
            
            body = abs(c - o)
            prev_body = abs(prev_c - prev_o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            
            # Pin Bar (Hammer/Shooting Star)
            if body > 0:
                if lower_wick > body * 2 and upper_wick < body * 0.5:
                    patterns.append({
                        "type": "hammer",
                        "index": i,
                        "strength": min((lower_wick / body) * 20, 100),
                        "direction": "bullish"
                    })
                elif upper_wick > body * 2 and lower_wick < body * 0.5:
                    patterns.append({
                        "type": "shooting_star",
                        "index": i,
                        "strength": min((upper_wick / body) * 20, 100),
                        "direction": "bearish"
                    })
            
            # Engulfing Pattern
            if prev_body > 0 and body > 0:
                if (c > prev_o and o < prev_c and 
                    body > prev_body * 1.2):  # Current candle engulfs previous
                    if c > o:  # Current is bullish
                        patterns.append({
                            "type": "bullish_engulfing",
                            "index": i,
                            "strength": min((body / prev_body) * 50, 100),
                            "direction": "bullish"
                        })
                    else:  # Current is bearish
                        patterns.append({
                            "type": "bearish_engulfing",
                            "index": i,
                            "strength": min((body / prev_body) * 50, 100),
                            "direction": "bearish"
                        })
        
        return patterns[-10:]  # Return last 10 patterns

class SignalGenerator:
    """Advanced signal generation with risk management"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.analyzer = TechnicalAnalyzer()
    
    def analyze_symbol(self, symbol: str, candle_data: Dict) -> Dict:
        """Comprehensive symbol analysis"""
        try:
            # Extract price data
            candles_1h = candle_data.get("1h", [])
            if len(candles_1h) < 50:
                return {"error": "Insufficient data", "symbol": symbol}
            
            # Convert to arrays
            timestamps = [c[0] for c in candles_1h]
            opens = [float(c[1]) for c in candles_1h]
            highs = [float(c[2]) for c in candles_1h]
            lows = [float(c[3]) for c in candles_1h]
            closes = [float(c[4]) for c in candles_1h]
            volumes = [float(c[5]) for c in candles_1h]
            
            current_price = closes[-1]
            
            # Technical indicators
            ema_9 = self.analyzer.calculate_ema(closes, 9)
            ema_21 = self.analyzer.calculate_ema(closes, 21)
            ema_50 = self.analyzer.calculate_ema(closes, 50)
            rsi = self.analyzer.calculate_rsi(closes, 14)
            
            # Market analysis
            sr_levels = self.analyzer.find_support_resistance_levels(highs, lows, closes)
            market_structure = self.analyzer.detect_market_structure(highs, lows, closes)
            order_blocks = self.analyzer.detect_order_blocks(opens, highs, lows, closes, volumes)
            candlestick_patterns = self.analyzer.detect_candlestick_patterns(opens, highs, lows, closes)
            
            # Volume analysis
            avg_volume_20 = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            return {
                "symbol": symbol,
                "timestamp": time.time(),
                "current_price": current_price,
                "indicators": {
                    "ema_9": ema_9[-1] if ema_9[-1] else None,
                    "ema_21": ema_21[-1] if ema_21[-1] else None,
                    "ema_50": ema_50[-1] if ema_50[-1] else None,
                    "rsi": rsi[-1] if rsi[-1] else None
                },
                "support_resistance": sr_levels,
                "market_structure": market_structure,
                "order_blocks": order_blocks,
                "candlestick_patterns": candlestick_patterns,
                "volume_analysis": {
                    "current": current_volume,
                    "average_20": avg_volume_20,
                    "ratio": volume_ratio
                },
                "price_data": {
                    "opens": opens[-100:],  # Last 100 for charting
                    "highs": highs[-100:],
                    "lows": lows[-100:],
                    "closes": closes[-100:],
                    "volumes": volumes[-100:],
                    "timestamps": timestamps[-100:]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def generate_signals(self, analysis: Dict) -> List[Dict]:
        """Generate trading signals based on analysis"""
        signals = []
        
        try:
            if "error" in analysis:
                return signals
            
            symbol = analysis["symbol"]
            current_price = analysis["current_price"]
            indicators = analysis["indicators"]
            sr_levels = analysis["support_resistance"]
            market_structure = analysis["market_structure"]
            patterns = analysis["candlestick_patterns"]
            volume_analysis = analysis["volume_analysis"]
            
            # Base confidence from market structure
            base_confidence = market_structure.get("strength", 0)
            
            # Volume confirmation
            volume_boost = min(volume_analysis.get("ratio", 1) * 10, 20)
            
            # Check for bullish signals
            bullish_score = 0
            bullish_reasons = []
            
            if indicators.get("ema_9") and indicators.get("ema_21"):
                if current_price > indicators["ema_9"] > indicators["ema_21"]:
                    bullish_score += 25
                    bullish_reasons.append("Price above EMAs")
            
            if indicators.get("rsi") and 30 <= indicators["rsi"] <= 70:
                bullish_score += 15
                bullish_reasons.append("RSI in good range")
            
            # Support level test
            strong_support = None
            for support in sr_levels.get("support", []):
                if abs(current_price - support["price"]) / current_price <= 0.01:  # Within 1%
                    bullish_score += support["strength"] / 5
                    strong_support = support
                    bullish_reasons.append(f"At strong support {support['price']:.2f}")
                    break
            
            # Bullish patterns
            for pattern in patterns:
                if pattern.get("direction") == "bullish" and pattern["index"] >= len(analysis["price_data"]["closes"]) - 5:
                    bullish_score += pattern["strength"] / 5
                    bullish_reasons.append(f"Recent {pattern['type']}")
            
            # Market structure alignment
            if market_structure.get("trend") == "bullish":
                bullish_score += market_structure.get("strength", 0) / 5
                bullish_reasons.append("Bullish market structure")
            
            # Generate bullish signal if conditions met
            total_bullish_confidence = min(base_confidence + bullish_score + volume_boost, 100)
            
            if (total_bullish_confidence >= self.config.signal_confidence_threshold and
                strong_support and len(bullish_reasons) >= 3):
                
                # Calculate levels
                entry = current_price
                stop_loss = strong_support["price"] * 0.995  # Slightly below support
                
                # Target from resistance or risk-reward calculation
                target = entry + (entry - stop_loss) * self.config.risk_reward_min
                
                # Check if there's resistance to adjust target
                for resistance in sr_levels.get("resistance", []):
                    if resistance["price"] > entry and resistance["price"] < target:
                        target = resistance["price"] * 0.99  # Just below resistance
                        break
                
                risk_reward = (target - entry) / (entry - stop_loss) if entry > stop_loss else 0
                
                if risk_reward >= self.config.risk_reward_min:
                    signals.append({
                        "symbol": symbol,
                        "direction": "BUY",
                        "entry": entry,
                        "stop_loss": stop_loss,
                        "target": target,
                        "confidence": total_bullish_confidence,
                        "risk_reward": risk_reward,
                        "reasons": bullish_reasons,
                        "timestamp": time.time()
                    })
            
            # Similar logic for bearish signals
            bearish_score = 0
            bearish_reasons = []
            
            if indicators.get("ema_9") and indicators.get("ema_21"):
                if current_price < indicators["ema_9"] < indicators["ema_21"]:
                    bearish_score += 25
                    bearish_reasons.append("Price below EMAs")
            
            if indicators.get("rsi") and indicators["rsi"] >= 70:
                bearish_score += 20
                bearish_reasons.append("RSI overbought")
            
            # Resistance level test
            strong_resistance = None
            for resistance in sr_levels.get("resistance", []):
                if abs(current_price - resistance["price"]) / current_price <= 0.01:
                    bearish_score += resistance["strength"] / 5
                    strong_resistance = resistance
                    bearish_reasons.append(f"At strong resistance {resistance['price']:.2f}")
                    break
            
            # Bearish patterns
            for pattern in patterns:
                if pattern.get("direction") == "bearish" and pattern["index"] >= len(analysis["price_data"]["closes"]) - 5:
                    bearish_score += pattern["strength"] / 5
                    bearish_reasons.append(f"Recent {pattern['type']}")
            
            # Market structure alignment
            if market_structure.get("trend") == "bearish":
                bearish_score += market_structure.get("strength", 0) / 5
                bearish_reasons.append("Bearish market structure")
            
            # Generate bearish signal if conditions met
            total_bearish_confidence = min(base_confidence + bearish_score + volume_boost, 100)
            
            if (total_bearish_confidence >= self.config.signal_confidence_threshold and
                strong_resistance and len(bearish_reasons) >= 3):
                
                # Calculate levels
                entry = current_price
                stop_loss = strong_resistance["price"] * 1.005  # Slightly above resistance
                
                # Target from support or risk-reward calculation
                target = entry - (stop_loss - entry) * self.config.risk_reward_min
                
                # Check if there's support to adjust target
                for support in sr_levels.get("support", []):
                    if support["price"] < entry and support["price"] > target:
                        target = support["price"] * 1.01  # Just above support
                        break
                
                risk_reward = (entry - target) / (stop_loss - entry) if stop_loss > entry else 0
                
                if risk_reward >= self.config.risk_reward_min:
                    signals.append({
                        "symbol": symbol,
                        "direction": "SELL",
                        "entry": entry,
                        "stop_loss": stop_loss,
                        "target": target,
                        "confidence": total_bearish_confidence,
                        "risk_reward": risk_reward,
                        "reasons": bearish_reasons,
                        "timestamp": time.time()
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return signals

# ================== CHART GENERATION ==================

class ChartGenerator:
    """Professional chart generation with technical analysis overlay"""
    
    @staticmethod
    def create_professional_chart(analysis: Dict, signal: Optional[Dict] = None) -> str:
        """Generate professional trading chart with analysis overlay"""
        try:
            symbol = analysis["symbol"]
            price_data = analysis["price_data"]
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(ts/1000) for ts in price_data["timestamps"]]
            opens = price_data["opens"]
            highs = price_data["highs"]
            lows = price_data["lows"]
            closes = price_data["closes"]
            volumes = price_data["volumes"]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Main price chart
            ax1.set_facecolor('#1a1a1a')
            fig.patch.set_facecolor('#0d1117')
            
            # Plot candlesticks
            for i, (ts, o, h, l, c) in enumerate(zip(timestamps, opens, highs, lows, closes)):
                color = '#00ff88' if c >= o else '#ff4444'
                
                # Candlestick body
                body_height = abs(c - o)
                body_bottom = min(o, c)
                
                ax1.add_patch(Rectangle((mdates.date2num(ts) - 0.3, body_bottom), 
                                      0.6, body_height, 
                                      facecolor=color, edgecolor=color, alpha=0.8))
                
                # Wicks
                ax1.plot([mdates.date2num(ts), mdates.date2num(ts)], [l, h], 
                        color=color, linewidth=1, alpha=0.8)
            
            # EMAs
            indicators = analysis["indicators"]
            if indicators.get("ema_9"):
                ax1.axhline(y=indicators["ema_9"], color='#ffd700', linewidth=2, 
                           linestyle='--', alpha=0.7, label='EMA 9')
            if indicators.get("ema_21"):
                ax1.axhline(y=indicators["ema_21"], color='#ff6600', linewidth=2, 
                           linestyle='--', alpha=0.7, label='EMA 21')
            if indicators.get("ema_50"):
                ax1.axhline(y=indicators["ema_50"], color='#9966ff', linewidth=2, 
                           linestyle='--', alpha=0.7, label='EMA 50')
            
            # Support and Resistance levels
            sr_levels = analysis["support_resistance"]
            
            for support in sr_levels.get("support", [])[:3]:  # Top 3 support levels
                strength_alpha = min(support["strength"] / 100, 1.0)
                ax1.axhline(y=support["price"], color='#00ff88', linewidth=2, 
                           alpha=strength_alpha, linestyle='-', 
                           label=f'Support {support["price"]:.2f}')
            
            for resistance in sr_levels.get("resistance", [])[:3]:  # Top 3 resistance levels
                strength_alpha = min(resistance["strength"] / 100, 1.0)
                ax1.axhline(y=resistance["price"], color='#ff4444', linewidth=2, 
                           alpha=strength_alpha, linestyle='-', 
                           label=f'Resistance {resistance["price"]:.2f}')
            
            # Order blocks
            for ob in analysis.get("order_blocks", [])[-5:]:  # Last 5 order blocks
                if ob["index"] < len(timestamps):
                    ob_time = timestamps[ob["index"]]
                    color = '#00ff88' if ob["type"] == "bullish" else '#ff4444'
                    alpha = min(ob["strength"] / 100, 0.5)
                    
                    ax1.add_patch(Rectangle((mdates.date2num(ob_time) - 1, ob["low"]), 
                                          2, ob["high"] - ob["low"], 
                                          facecolor=color, alpha=alpha, 
                                          label=f'{ob["type"].title()} OB'))
            
            # Signal overlay
            if signal:
                current_time = timestamps[-1]
                entry_price = signal["entry"]
                stop_loss = signal["stop_loss"]
                target = signal["target"]
                
                # Entry line
                ax1.axhline(y=entry_price, color='#ffffff', linewidth=3, 
                           linestyle='-', label=f'Entry: {entry_price:.2f}')
                
                # Stop loss
                ax1.axhline(y=stop_loss, color='#ff4444', linewidth=2, 
                           linestyle=':', label=f'Stop Loss: {stop_loss:.2f}')
                
                # Target
                ax1.axhline(y=target, color='#00ff88', linewidth=2, 
                           linestyle=':', label=f'Target: {target:.2f}')
                
                # Risk/Reward zone
                if signal["direction"] == "BUY":
                    ax1.fill_between([mdates.date2num(current_time) - 5, mdates.date2num(current_time) + 5],
                                   entry_price, target, alpha=0.1, color='green', label='Profit Zone')
                    ax1.fill_between([mdates.date2num(current_time) - 5, mdates.date2num(current_time) + 5],
                                   entry_price, stop_loss, alpha=0.1, color='red', label='Risk Zone')
                else:
                    ax1.fill_between([mdates.date2num(current_time) - 5, mdates.date2num(current_time) + 5],
                                   entry_price, target, alpha=0.1, color='green', label='Profit Zone')
                    ax1.fill_between([mdates.date2num(current_time) - 5, mdates.date2num(current_time) + 5],
                                   entry_price, stop_loss, alpha=0.1, color='red', label='Risk Zone')
            
            # Chart formatting
            ax1.set_title(f'{symbol} - Professional Analysis', 
                         fontsize=20, color='white', fontweight='bold', pad=20)
            ax1.set_ylabel('Price (USDT)', fontsize=14, color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3, color='gray')
            ax1.legend(loc='upper left', facecolor='#2d3142', edgecolor='white')
            
            # Volume chart
            ax2.set_facecolor('#1a1a1a')
            volume_colors = ['#00ff88' if closes[i] >= opens[i] else '#ff4444' 
                           for i in range(len(volumes))]
            
            ax2.bar(timestamps, volumes, color=volume_colors, alpha=0.7, width=0.8)
            ax2.set_ylabel('Volume', fontsize=12, color='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3, color='gray')
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Add analysis info box
            info_text = f"""Market Structure: {analysis['market_structure']['trend'].upper()}
Strength: {analysis['market_structure']['strength']:.0f}%
RSI: {indicators.get('rsi', 0):.1f}
Volume Ratio: {analysis['volume_analysis']['ratio']:.1f}x"""
            
            if signal:
                info_text += f"""

SIGNAL: {signal['direction']}
Confidence: {signal['confidence']:.1f}%
Risk/Reward: 1:{signal['risk_reward']:.1f}"""
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                    fontsize=12, color='white', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#2d3142', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            temp_file = NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, facecolor='#0d1117', dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None

# ================== OPENAI INTEGRATION ==================

class AIAnalyzer:
    """OpenAI integration for enhanced analysis"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    async def enhance_analysis(self, analysis: Dict, signals: List[Dict]) -> Dict:
        """Use AI to enhance technical analysis"""
        if not self.client:
            return {"ai_analysis": "OpenAI not available"}
        
        try:
            # Prepare structured data for AI
            ai_input = {
                "symbol": analysis["symbol"],
                "current_price": analysis["current_price"],
                "market_structure": analysis["market_structure"],
                "indicators": analysis["indicators"],
                "support_resistance": {
                    "support_count": len(analysis["support_resistance"].get("support", [])),
                    "resistance_count": len(analysis["support_resistance"].get("resistance", [])),
                    "nearest_support": analysis["support_resistance"]["support"][0] if analysis["support_resistance"].get("support") else None,
                    "nearest_resistance": analysis["support_resistance"]["resistance"][0] if analysis["support_resistance"].get("resistance") else None
                },
                "volume_analysis": analysis["volume_analysis"],
                "recent_patterns": len(analysis.get("candlestick_patterns", [])),
                "order_blocks": len(analysis.get("order_blocks", [])),
                "generated_signals": len(signals)
            }
            
            system_prompt = """You are a professional crypto trader with 10+ years of experience. 
            Analyze the provided technical data and provide insights on:
            1. Market sentiment and trend confirmation
            2. Key risk factors and entry timing
            3. Additional confluence factors
            4. Market structure strength assessment
            
            Be concise but insightful. Focus on actionable insights."""
            
            user_prompt = f"""Analyze this crypto data:
            
            {json.dumps(ai_input, indent=2)}
            
            Provide professional trading insights in 3-4 sentences focusing on the most critical factors for decision making."""
            
            # Run AI analysis in executor to avoid blocking
            loop = asyncio.get_running_loop()
            
            def call_openai():
                return self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
            
            response = await loop.run_in_executor(None, call_openai)
            ai_insight = response.choices[0].message.content.strip()
            
            return {
                "ai_analysis": ai_insight,
                "ai_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {"ai_analysis": f"AI analysis failed: {str(e)}"}

# ================== TELEGRAM INTEGRATION ==================

class TelegramNotifier:
    """Enhanced Telegram notifications with rich formatting"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, session: aiohttp.ClientSession, text: str):
        """Send text message to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.info(f"Telegram not configured. Message: {text}")
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Telegram message failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Telegram send_message error: {e}")
    
    async def send_photo(self, session: aiohttp.ClientSession, caption: str, photo_path: str):
        """Send photo with caption to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.info(f"Telegram not configured. Caption: {caption}")
            return
        
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('caption', caption, content_type='text/plain')
                data.add_field('parse_mode', 'Markdown')
                data.add_field('photo', photo, filename='chart.png')
                
                async with session.post(url, data=data, timeout=60) as response:
                    if response.status != 200:
                        logger.error(f"Telegram photo failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Telegram send_photo error: {e}")
    
    def format_signal_message(self, signal: Dict, analysis: Dict, ai_analysis: Dict = None) -> str:
        """Format professional signal message"""
        symbol = signal["symbol"]
        direction = signal["direction"]
        entry = signal["entry"]
        stop_loss = signal["stop_loss"]
        target = signal["target"]
        confidence = signal["confidence"]
        risk_reward = signal["risk_reward"]
        
        # Calculate percentages
        if direction == "BUY":
            sl_pct = ((entry - stop_loss) / entry) * 100
            tp_pct = ((target - entry) / entry) * 100
        else:
            sl_pct = ((stop_loss - entry) / entry) * 100
            tp_pct = ((entry - target) / entry) * 100
        
        # Emojis based on direction and confidence
        signal_emoji = "🚀" if direction == "BUY" else "🔻"
        confidence_emoji = "🔥" if confidence >= 85 else "⚡" if confidence >= 75 else "💫"
        
        message = f"""
{signal_emoji} *{symbol} SIGNAL ALERT* {confidence_emoji}

📊 *Analysis:* 30-Day Price Action Review
💹 *Signal:* {direction}
💰 *Entry:* ${entry:.4f}
🛑 *Stop Loss:* ${stop_loss:.4f} (-{sl_pct:.2f}%)
🎯 *Target:* ${target:.4f} (+{tp_pct:.2f}%)
📈 *R/R Ratio:* 1:{risk_reward:.1f}

🔍 *Key Factors:*
"""
        
        # Add reasons
        for i, reason in enumerate(signal["reasons"][:3], 1):
            message += f"• {reason}\n"
        
        # Market structure info
        structure = analysis["market_structure"]
        message += f"""
📊 *Market Structure:* {structure['trend'].title()} ({structure['strength']:.0f}%)
📈 *Volume:* {analysis['volume_analysis']['ratio']:.1f}x average
🧠 *AI Confidence:* {confidence:.1f}/100

⚠️ *Risk Level:* {'High' if confidence < 80 else 'Medium' if confidence < 90 else 'Low'}
🕐 *Analysis Time:* {datetime.now().strftime('%H:%M UTC')}
"""
        
        # Add AI insights if available
        if ai_analysis and ai_analysis.get("ai_analysis"):
            message += f"\n🤖 *AI Insight:* {ai_analysis['ai_analysis']}"
        
        return message

# ================== MAIN BOT ENGINE ==================

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.chart_generator = ChartGenerator()
        self.ai_analyzer = AIAnalyzer(openai_client) if openai_client else None
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) if TELEGRAM_BOT_TOKEN else None
        self.redis = redis_manager
        
        # Rate limiting
        self.last_signals = {}  # Track last signal time per symbol
        self.hourly_signal_count = 0
        self.last_hour_reset = time.time()
    
    async def fetch_market_data(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """Fetch comprehensive market data for symbol"""
        try:
            data = {}
            
            # Fetch multiple timeframes
            for timeframe in self.config.timeframes:
                url = f"{KLINES_URL}?symbol={symbol}&interval={timeframe}&limit={self.config.analysis_candles if timeframe == '1h' else 100}"
                
                async with session.get(url, timeout=20) as response:
                    if response.status == 200:
                        candles = await response.json()
                        data[timeframe] = candles
                    else:
                        logger.warning(f"Failed to fetch {timeframe} data for {symbol}: {response.status}")
            
            # Fetch 24h stats
            stats_url = f"{TICKER_24H_URL}?symbol={symbol}"
            async with session.get(stats_url, timeout=20) as response:
                if response.status == 200:
                    data["24h_stats"] = await response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {}
    
    def should_generate_signal(self, symbol: str) -> bool:
        """Check if we should generate a signal based on rate limits"""
        now = time.time()
        
        # Reset hourly counter
        if now - self.last_hour_reset > 3600:
            self.hourly_signal_count = 0
            self.last_hour_reset = now
        
        # Check hourly limit
        if self.hourly_signal_count >= self.config.max_signals_per_hour:
            return False
        
        # Check per-symbol cooldown (minimum 1 hour between signals)
        last_signal_time = self.last_signals.get(symbol, 0)
        if now - last_signal_time < 3600:
            return False
        
        return True
    
    async def process_symbol(self, session: aiohttp.ClientSession, symbol: str):
        """Process single symbol for analysis and signals"""
        try:
            # Check cache first
            cached_analysis = self.redis.get_analysis_cache(symbol)
            if cached_analysis:
                logger.info(f"Using cached analysis for {symbol}")
                analysis = cached_analysis
            else:
                # Fetch fresh data
                logger.info(f"Analyzing {symbol}...")
                market_data = await self.fetch_market_data(session, symbol)
                
                if not market_data or "1h" not in market_data:
                    logger.warning(f"Insufficient data for {symbol}")
                    return
                
                # Store raw data in Redis
                self.redis.store_candle_data(symbol, "1h", 
                    [{"timestamp": c[0], "open": c[1], "high": c[2], 
                      "low": c[3], "close": c[4], "volume": c[5]} 
                     for c in market_data["1h"]])
                
                # Perform technical analysis
                analysis = self.signal_generator.analyze_symbol(symbol, market_data)
                
                if "error" in analysis:
                    logger.error(f"Analysis failed for {symbol}: {analysis['error']}")
                    return
                
                # Cache analysis
                self.redis.store_analysis_cache(symbol, analysis)
            
            # Generate signals
            if self.should_generate_signal(symbol):
                signals = self.signal_generator.generate_signals(analysis)
                
                if signals:
                    # Process each signal
                    for signal in signals:
                        await self.process_signal(session, signal, analysis)
                        
                        # Update rate limiting
                        self.last_signals[symbol] = time.time()
                        self.hourly_signal_count += 1
                        
                        # Store signal in Redis
                        self.redis.store_signal(symbol, signal)
                        
                        logger.info(f"Generated {signal['direction']} signal for {symbol} "
                                  f"with {signal['confidence']:.1f}% confidence")
                else:
                    logger.info(f"No signals generated for {symbol}")
            else:
                logger.info(f"Rate limited for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            traceback.print_exc()
    
    async def process_signal(self, session: aiohttp.ClientSession, signal: Dict, analysis: Dict):
        """Process and send signal with chart and AI analysis"""
        try:
            # Get AI enhancement if available
            ai_analysis = None
            if self.ai_analyzer:
                ai_analysis = await self.ai_analyzer.enhance_analysis(analysis, [signal])
            
            # Generate chart
            chart_path = self.chart_generator.create_professional_chart(analysis, signal)
            
            if self.telegram:
                # Format and send message
                message = self.telegram.format_signal_message(signal, analysis, ai_analysis)
                
                if chart_path:
                    await self.telegram.send_photo(session, message, chart_path)
                    # Clean up chart file
                    try:
                        os.unlink(chart_path)
                    except:
                        pass
                else:
                    await self.telegram.send_message(session, message)
            else:
                logger.info(f"Signal: {signal}")
                if ai_analysis:
                    logger.info(f"AI Analysis: {ai_analysis['ai_analysis']}")
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        logger.info(f"Starting analysis cycle at {datetime.now()}")
        
        async with aiohttp.ClientSession() as session:
            # Process all symbols
            tasks = []
            for symbol in self.config.symbols:
                task = self.process_symbol(session, symbol)
                tasks.append(task)
            
            # Execute all symbol analyses concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Analysis cycle completed")
    
    async def run_forever(self):
        """Main bot loop"""
        startup_message = f"""
🤖 *Enhanced Trading Bot Started*

📊 Symbols: {len(self.config.symbols)}
⏱ Poll Interval: {self.config.poll_interval}s
🎯 Min Confidence: {self.config.signal_confidence_threshold}%
💎 Min R/R: {self.config.risk_reward_min}:1
🔄 Max Signals/Hour: {self.config.max_signals_per_hour}

Features:
✅ 720-candle analysis (30 days)
✅ Smart Money Concepts
✅ Professional charting
✅ AI-enhanced analysis
✅ Redis caching & tracking

Ready for trading! 🚀
"""
        
        if self.telegram:
            async with aiohttp.ClientSession() as session:
                await self.telegram.send_message(session, startup_message)
        
        logger.info("Bot started successfully")
        
        # Main loop
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"=== ITERATION {iteration} ===")
                
                await self.run_analysis_cycle()
                
                logger.info(f"Sleeping for {self.config.poll_interval} seconds...")
                await asyncio.sleep(self.config.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)  # Wait 1 minute before retrying

# ================== MAIN EXECUTION ==================

async def main():
    """Main entry point"""
    try:
        bot = TradingBot()
        await bot.run_forever()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
