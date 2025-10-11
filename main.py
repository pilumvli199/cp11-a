#!/usr/bin/env python3
# main.py - Qwen 2.5 VL 72B Crypto Trading Bot (OpenRouter)
# Charts analysis every 1 hour with rate limit handling (10 sec per scan)

import os
import asyncio
import aiohttp
import traceback
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import io
from PIL import Image

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AAVEUSDT",
    "TRXUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LTCUSDT"
]

POLL_INTERVAL = 3600  # 1 hour = 3600 seconds
SCAN_DELAY = 10  # 10 seconds delay between each coin scan (rate limit safety)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2-vl-72b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 70.0))

# Binance API - 1H timeframe, last 100 candles (enough for analysis)
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"

# ---------------- Utils ----------------

def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"

# ---------------- Chart Generation ----------------

def generate_chart_image(symbol, candles):
    """
    Generate candlestick chart from Binance candle data
    Returns: bytes of PNG image
    """
    try:
        if not candles or len(candles) < 20:
            # Fallback: simple text image
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f"{symbol}\nInsufficient Data", 
                    ha='center', va='center', fontsize=20)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        # Parse candle data
        times = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for candle in candles:
            try:
                timestamp = int(candle[0])
                times.append(datetime.utcfromtimestamp(timestamp / 1000))
                opens.append(float(candle[1]))
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
                volumes.append(float(candle[5]))
            except Exception:
                continue

        if len(closes) < 20:
            raise Exception("Not enough valid candles")

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlesticks
        x = date2num(times)
        for i in range(len(x)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            # High-low line
            ax1.plot([x[i], x[i]], [lows[i], highs[i]], 
                    color=color, linewidth=1.0, alpha=0.8)
            # Open-close body
            ax1.plot([x[i], x[i]], [opens[i], closes[i]], 
                    color=color, linewidth=4.0, solid_capstyle='round')

        # Calculate and plot simple moving averages
        def sma(data, period):
            result = []
            for i in range(len(data)):
                if i < period - 1:
                    result.append(None)
                else:
                    result.append(sum(data[i-period+1:i+1]) / period)
            return result

        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)

        # Plot SMAs
        valid_sma20 = [(x[i], sma20[i]) for i in range(len(sma20)) if sma20[i] is not None]
        valid_sma50 = [(x[i], sma50[i]) for i in range(len(sma50)) if sma50[i] is not None]
        
        if valid_sma20:
            ax1.plot([v[0] for v in valid_sma20], [v[1] for v in valid_sma20], 
                    label='SMA20', linewidth=1.5, color='blue', alpha=0.7)
        if valid_sma50:
            ax1.plot([v[0] for v in valid_sma50], [v[1] for v in valid_sma50], 
                    label='SMA50', linewidth=1.5, color='orange', alpha=0.7)

        ax1.set_title(f"{symbol} - 1H Chart", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)

        # Volume subplot
        colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(closes))]
        ax2.bar(x, volumes, color=colors, alpha=0.6, width=0.0008)
        ax2.set_title('Volume', fontsize=10)
        ax2.set_ylabel('Volume', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()

    except Exception as e:
        print(f"Chart generation error for {symbol}: {e}")
        traceback.print_exc()
        # Return simple error image
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, f"{symbol}\nChart Error: {str(e)[:50]}", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

# ---------------- Qwen Vision Analysis ----------------

async def analyze_chart_with_qwen(session, image_bytes, symbol, current_price):
    """
    Send chart image to Qwen 2.5 VL 72B for analysis via OpenRouter
    """
    if not OPENROUTER_API_KEY:
        return {
            "side": "none",
            "confidence": 0,
            "reason": "OpenRouter API key not configured",
            "analysis": "API key missing"
        }

    try:
        # Convert image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://crypto-scanner.local",
            "X-Title": "Crypto Chart Analyzer"
        }

        prompt = f"""Analyze this {symbol} 1-hour cryptocurrency chart for SHORT-TERM/SCALPING trading.

Current Price: {fmt_price(current_price)}

⚡ PROVIDE PRECISE ANALYSIS:

1. **TREND DIRECTION**: Bullish/Bearish/Sideways? (be specific)
2. **SUPPORT LEVELS**: Identify key support levels visible on chart
3. **RESISTANCE LEVELS**: Identify key resistance levels visible on chart
4. **CANDLESTICK PATTERNS**: Any significant patterns? (Doji, Hammer, Engulfing, etc.)
5. **CHART PATTERNS**: Head & Shoulders, Triangle, Flag, Wedge? 
6. **MOVING AVERAGES**: Price position relative to MAs? Crossovers?
7. **VOLUME ANALYSIS**: Volume increasing/decreasing? Confirmation?
8. **MOMENTUM**: Is momentum bullish or bearish?

🎯 TRADING SIGNAL (CRITICAL):
- **ACTION**: BUY / SELL / HOLD (choose one)
- **CONFIDENCE**: X/10 (be honest, not always high)
- **ENTRY PRICE**: Specific price level
- **STOP LOSS**: Where to place SL (critical for risk management)
- **TAKE PROFIT 1**: First target
- **TAKE PROFIT 2**: Second target (if applicable)
- **TIMEFRAME**: Expected trade duration (15min, 1H, 4H, etc.)

⚠️ IMPORTANT:
- Only give BUY/SELL signal if confidence >= 7/10
- Be precise with entry/SL/TP levels
- Consider risk/reward ratio (minimum 1:1.5)
- FALSE SIGNALS CAUSE LOSSES - be accurate!

Format your response clearly with each point."""

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.2,  # Low for precise analysis
            "max_tokens": 1500,
            "top_p": 0.9
        }

        # Make async API call
        async with session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ OpenRouter API error {response.status}: {error_text[:200]}")
                return {
                    "side": "none",
                    "confidence": 0,
                    "reason": f"API error: {response.status}",
                    "analysis": error_text[:200]
                }

            result = await response.json()
            
            # Extract analysis
            analysis_text = result['choices'][0]['message']['content']
            
            # Parse the analysis to extract signal
            signal_data = parse_qwen_analysis(analysis_text, current_price)
            signal_data['analysis'] = analysis_text
            
            # Check token usage
            usage = result.get('usage', {})
            print(f"   Tokens used: {usage.get('total_tokens', 'N/A')}")
            
            return signal_data

    except asyncio.TimeoutError:
        print(f"⏱️ Timeout analyzing {symbol}")
        return {
            "side": "none",
            "confidence": 0,
            "reason": "API timeout",
            "analysis": "Request timed out"
        }
    except Exception as e:
        print(f"❌ Qwen analysis error for {symbol}: {e}")
        traceback.print_exc()
        return {
            "side": "none",
            "confidence": 0,
            "reason": f"Analysis error: {str(e)[:50]}",
            "analysis": str(e)
        }

def parse_qwen_analysis(text, current_price):
    """
    Parse Qwen's text response to extract trading signal
    """
    try:
        text_upper = text.upper()
        
        # Detect signal
        side = "none"
        if "ACTION: BUY" in text_upper or "ACTION**: BUY" in text_upper:
            side = "BUY"
        elif "ACTION: SELL" in text_upper or "ACTION**: SELL" in text_upper:
            side = "SELL"
        elif "BUY SIGNAL" in text_upper and "CONFIDENCE" in text_upper:
            side = "BUY"
        elif "SELL SIGNAL" in text_upper and "CONFIDENCE" in text_upper:
            side = "SELL"
        
        # Extract confidence (X/10 format)
        confidence = 0
        import re
        conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)/10', text_upper)
        if conf_match:
            confidence = int(conf_match.group(1)) * 10  # Convert to percentage
        else:
            # Try percentage format
            conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)%', text_upper)
            if conf_match:
                confidence = int(conf_match.group(1))
        
        # Extract entry price
        entry = current_price
        entry_match = re.search(r'ENTRY[^\d]*(\d+\.?\d*)', text)
        if entry_match:
            try:
                entry = float(entry_match.group(1))
            except:
                pass
        
        # Extract stop loss
        sl = entry * 0.98 if side == "BUY" else entry * 1.02  # Default 2%
        sl_match = re.search(r'STOP LOSS[^\d]*(\d+\.?\d*)', text)
        if sl_match:
            try:
                sl = float(sl_match.group(1))
            except:
                pass
        
        # Extract take profit
        tp = entry * 1.03 if side == "BUY" else entry * 0.97  # Default 3%
        tp_match = re.search(r'TAKE PROFIT[^\d]*1?[^\d]*(\d+\.?\d*)', text)
        if tp_match:
            try:
                tp = float(tp_match.group(1))
            except:
                pass
        
        # Calculate R:R
        rr = 0
        if side == "BUY" and (entry - sl) > 0:
            rr = (tp - entry) / (entry - sl)
        elif side == "SELL" and (sl - entry) > 0:
            rr = (entry - tp) / (sl - entry)
        
        # Extract reason (first few lines)
        reason_lines = text.split('\n')[:3]
        reason = ' '.join([line.strip() for line in reason_lines if line.strip()])[:150]
        
        return {
            "side": side,
            "confidence": confidence,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": round(rr, 2),
            "reason": reason
        }
    
    except Exception as e:
        print(f"Parse error: {e}")
        return {
            "side": "none",
            "confidence": 0,
            "entry": current_price,
            "sl": 0,
            "tp": 0,
            "rr": 0,
            "reason": "Failed to parse analysis"
        }

# ---------------- Fetch Data ----------------

async def fetch_json(session, url):
    """Fetch JSON from URL with error handling"""
    try:
        async with session.get(url, timeout=30) as r:
            if r.status != 200:
                text = await r.text()
                print(f"   Fetch error {r.status}: {text[:100]}")
                return None
            return await r.json()
    except Exception as e:
        print(f"   Fetch exception: {e}")
        return None

# ---------------- Telegram ----------------

async def send_text(session, text):
    """Send text message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}) as resp:
            if resp.status != 200:
                print(f"Telegram text error: {resp.status}")
    except Exception as e:
        print(f"Telegram text exception: {e}")

async def send_photo_bytes(session, caption, image_bytes):
    """Send photo from bytes to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        data = aiohttp.FormData()
        data.add_field("chat_id", TELEGRAM_CHAT_ID)
        data.add_field("caption", caption[:1024])  # Telegram caption limit
        data.add_field("photo", image_bytes, filename="chart.png", content_type="image/png")
        
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Telegram photo error {resp.status}: {error_text[:200]}")
    except Exception as e:
        print(f"Telegram photo exception: {e}")

# ---------------- Main Loop ----------------

async def main_scanner_loop():
    """Main scanning loop - runs every 1 hour"""
    
    async with aiohttp.ClientSession() as session:
        startup_msg = f"""🤖 Qwen VL Crypto Scanner Started!

📊 Symbols: {len(SYMBOLS)} coins
⏰ Scan Interval: {POLL_INTERVAL//60} minutes
🔄 Rate Limit: {SCAN_DELAY} sec between scans
🤖 Model: {OPENROUTER_MODEL}
🎯 Confidence Threshold: {SIGNAL_CONF_THRESHOLD}%

Starting first scan..."""
        
        print(startup_msg)
        await send_text(session, startup_msg)
        
        iteration = 0
        
        while True:
            iteration += 1
            scan_start = datetime.utcnow()
            
            print(f"\n{'='*70}")
            print(f"🔍 SCAN #{iteration} @ {scan_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*70}")
            
            signals_found = 0
            scan_results = []
            
            for idx, symbol in enumerate(SYMBOLS, 1):
                try:
                    print(f"\n[{idx}/{len(SYMBOLS)}] 📊 Analyzing {symbol}...")
                    
                    # Fetch candle data
                    candles = await fetch_json(session, CANDLE_URL.format(symbol=symbol))
                    
                    if not candles or len(candles) < 20:
                        print(f"   ❌ Insufficient data for {symbol}")
                        continue
                    
                    # Get current price
                    current_price = float(candles[-1][4])  # Close price of last candle
                    print(f"   💰 Current price: {fmt_price(current_price)}")
                    
                    # Generate chart
                    print(f"   📈 Generating chart...")
                    chart_bytes = generate_chart_image(symbol, candles)
                    
                    # Analyze with Qwen Vision
                    print(f"   🤖 Analyzing with Qwen VL...")
                    signal = await analyze_chart_with_qwen(session, chart_bytes, symbol, current_price)
                    
                    # Check if signal meets threshold
                    if signal['side'] != 'none' and signal['confidence'] >= SIGNAL_CONF_THRESHOLD:
                        signals_found += 1
                        
                        # Prepare message
                        msg = f"""🚀 {symbol} {signal['side']} SIGNAL

📈 Entry: {fmt_price(signal['entry'])}
🛑 Stop Loss: {fmt_price(signal['sl'])}
🎯 Take Profit: {fmt_price(signal['tp'])}
📊 Confidence: {signal['confidence']}%
⚖️ Risk/Reward: 1:{signal['rr']}

💡 Analysis:
{signal.get('reason', '')[:200]}

🤖 Model: Qwen 2.5 VL 72B"""
                        
                        # Send to Telegram with chart
                        await send_photo_bytes(session, msg, chart_bytes)
                        
                        print(f"   ✅ SIGNAL SENT: {signal['side']} | Conf: {signal['confidence']}%")
                        
                        scan_results.append({
                            "symbol": symbol,
                            "signal": signal['side'],
                            "confidence": signal['confidence']
                        })
                    else:
                        print(f"   📉 No signal (Conf: {signal['confidence']}% | {signal['side']})")
                        print(f"      Reason: {signal.get('reason', 'N/A')[:80]}")
                    
                    # Rate limit delay (10 seconds between scans)
                    if idx < len(SYMBOLS):
                        print(f"   ⏳ Waiting {SCAN_DELAY}s (rate limit)...")
                        await asyncio.sleep(SCAN_DELAY)
                
                except Exception as e:
                    print(f"   ❌ Error analyzing {symbol}: {e}")
                    traceback.print_exc()
                    continue
            
            # Scan summary
            scan_duration = (datetime.utcnow() - scan_start).total_seconds()
            
            summary = f"""📊 Scan #{iteration} Complete

✅ Analyzed: {len(SYMBOLS)} symbols
🎯 Signals Found: {signals_found}
⏱️ Duration: {scan_duration:.1f}s
🔄 Next scan: {POLL_INTERVAL//60} minutes

{'─'*40}"""
            
            if signals_found > 0:
                summary += "\n🚀 Signals:\n"
                for result in scan_results:
                    summary += f"• {result['symbol']}: {result['signal']} ({result['confidence']}%)\n"
            
            print(f"\n{summary}")
            await send_text(session, summary)
            
            # Wait for next scan
            print(f"\n💤 Sleeping for {POLL_INTERVAL//60} minutes...")
            await asyncio.sleep(POLL_INTERVAL)

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    try:
        print("🚀 Starting Qwen VL Crypto Scanner...")
        asyncio.run(main_scanner_loop())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
