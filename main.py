#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Exchange Crypto Data Fetcher - Testing Version
Fetches data every 5 minutes and sends to Telegram
"""

import os
import time
import hmac
import hashlib
import asyncio
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import schedule
import logging
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DELTA_API_KEY = os.getenv('DELTA_API_KEY')
DELTA_API_SECRET = os.getenv('DELTA_API_SECRET')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False') == 'True'

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# EXCHANGE DATA FETCHERS
# =============================================================================

class BinanceFetcher:
    """Fetch Binance spot + futures data (Public API)"""
    BASE_URL = "https://api.binance.com"
    FAPI_URL = "https://fapi.binance.com"
    
    @staticmethod
    def get_btc_data() -> Dict:
        try:
            # Spot Volume
            spot_response = requests.get(
                f"{BinanceFetcher.BASE_URL}/api/v3/ticker/24hr",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            spot_data = spot_response.json()
            
            # Futures Open Interest
            oi_response = requests.get(
                f"{BinanceFetcher.FAPI_URL}/fapi/v1/openInterest",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            oi_data = oi_response.json()
            
            # Funding Rate
            funding_response = requests.get(
                f"{BinanceFetcher.FAPI_URL}/fapi/v1/fundingRate",
                params={"symbol": "BTCUSDT", "limit": 1},
                timeout=10
            )
            funding_data = funding_response.json()[0] if funding_response.json() else {}
            
            return {
                "exchange": "Binance",
                "spot_volume_24h": float(spot_data.get('quoteVolume', 0)),
                "spot_price": float(spot_data.get('lastPrice', 0)),
                "futures_oi": float(oi_data.get('openInterest', 0)),
                "funding_rate": float(funding_data.get('fundingRate', 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
            return {"exchange": "Binance", "error": str(e)}


class OKXFetcher:
    """Fetch OKX derivatives data (Public API)"""
    BASE_URL = "https://www.okx.com"
    
    @staticmethod
    def get_btc_data() -> Dict:
        try:
            # Spot ticker
            ticker_response = requests.get(
                f"{OKXFetcher.BASE_URL}/api/v5/market/ticker",
                params={"instId": "BTC-USDT"},
                timeout=10
            )
            ticker_data = ticker_response.json()['data'][0]
            
            # SWAP Open Interest
            oi_response = requests.get(
                f"{OKXFetcher.BASE_URL}/api/v5/public/open-interest",
                params={"instId": "BTC-USDT-SWAP"},
                timeout=10
            )
            oi_data = oi_response.json()['data'][0] if oi_response.json()['data'] else {}
            
            # Funding Rate
            funding_response = requests.get(
                f"{OKXFetcher.BASE_URL}/api/v5/public/funding-rate",
                params={"instId": "BTC-USDT-SWAP"},
                timeout=10
            )
            funding_data = funding_response.json()['data'][0] if funding_response.json()['data'] else {}
            
            return {
                "exchange": "OKX",
                "spot_volume_24h": float(ticker_data.get('volCcy24h', 0)),
                "spot_price": float(ticker_data.get('last', 0)),
                "swap_oi": float(oi_data.get('oi', 0)),
                "funding_rate": float(funding_data.get('fundingRate', 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"OKX fetch error: {e}")
            return {"exchange": "OKX", "error": str(e)}


class BybitFetcher:
    """Fetch Bybit perpetuals data (Public API)"""
    BASE_URL = "https://api.bybit.com"
    
    @staticmethod
    def get_btc_data() -> Dict:
        try:
            # Ticker data
            ticker_response = requests.get(
                f"{BybitFetcher.BASE_URL}/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"},
                timeout=10
            )
            ticker_data = ticker_response.json()['result']['list'][0]
            
            # Open Interest
            oi_response = requests.get(
                f"{BybitFetcher.BASE_URL}/v5/market/open-interest",
                params={"category": "linear", "symbol": "BTCUSDT", "intervalTime": "5min"},
                timeout=10
            )
            oi_data = oi_response.json()['result']['list'][0] if oi_response.json()['result']['list'] else {}
            
            # Funding Rate
            funding_response = requests.get(
                f"{BybitFetcher.BASE_URL}/v5/market/funding/history",
                params={"category": "linear", "symbol": "BTCUSDT", "limit": 1},
                timeout=10
            )
            funding_data = funding_response.json()['result']['list'][0] if funding_response.json()['result']['list'] else {}
            
            return {
                "exchange": "Bybit",
                "spot_volume_24h": float(ticker_data.get('turnover24h', 0)),
                "spot_price": float(ticker_data.get('lastPrice', 0)),
                "perpetual_oi": float(oi_data.get('openInterest', 0)),
                "funding_rate": float(funding_data.get('fundingRate', 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Bybit fetch error: {e}")
            return {"exchange": "Bybit", "error": str(e)}


class DeribitFetcher:
    """Fetch Deribit options data (Public API)"""
    BASE_URL = "https://www.deribit.com/api/v2"
    
    @staticmethod
    def get_btc_data() -> Dict:
        try:
            # Options book summary
            summary_response = requests.get(
                f"{DeribitFetcher.BASE_URL}/public/get_book_summary_by_currency",
                params={"currency": "BTC", "kind": "option"},
                timeout=10
            )
            summary_data = summary_response.json()['result']
            
            # Calculate total Call and Put OI
            total_call_oi = sum(float(item['open_interest']) for item in summary_data if item['instrument_name'].endswith('C'))
            total_put_oi = sum(float(item['open_interest']) for item in summary_data if item['instrument_name'].endswith('P'))
            
            # PCR Ratio
            pcr_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Spot price from futures
            futures_response = requests.get(
                f"{DeribitFetcher.BASE_URL}/public/ticker",
                params={"instrument_name": "BTC-PERPETUAL"},
                timeout=10
            )
            futures_data = futures_response.json()['result']
            
            return {
                "exchange": "Deribit",
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "pcr_ratio": round(pcr_ratio, 2),
                "spot_price": float(futures_data.get('last_price', 0)),
                "volume_24h": float(futures_data.get('stats', {}).get('volume', 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Deribit fetch error: {e}")
            return {"exchange": "Deribit", "error": str(e)}


class DeltaExchangeFetcher:
    """Fetch Delta Exchange data (Authenticated API)"""
    BASE_URL = "https://api.delta.exchange"
    
    @staticmethod
    def _generate_signature(method: str, endpoint: str, payload: str = "") -> Dict[str, str]:
        """Generate HMAC signature for Delta Exchange"""
        timestamp = str(int(time.time()))
        message = method + timestamp + endpoint + payload
        signature = hmac.new(
            DELTA_API_SECRET.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "api-key": DELTA_API_KEY,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def get_btc_data() -> Dict:
        try:
            # Get tickers
            endpoint = "/v2/tickers"
            headers = DeltaExchangeFetcher._generate_signature("GET", endpoint)
            
            response = requests.get(
                f"{DeltaExchangeFetcher.BASE_URL}{endpoint}",
                headers=headers,
                timeout=10
            )
            
            tickers = response.json()['result']
            
            # Find BTC-USDT
            btc_ticker = next((t for t in tickers if 'BTCUSDT' in t['symbol']), None)
            
            if not btc_ticker:
                return {"exchange": "Delta Exchange", "error": "BTC ticker not found"}
            
            return {
                "exchange": "Delta Exchange",
                "spot_price": float(btc_ticker.get('close', 0)),
                "volume_24h": float(btc_ticker.get('volume', 0)),
                "oi": float(btc_ticker.get('open_interest', 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Delta Exchange fetch error: {e}")
            return {"exchange": "Delta Exchange", "error": str(e)}


# =============================================================================
# TELEGRAM SENDER
# =============================================================================

class TelegramSender:
    """Send formatted messages to Telegram"""
    
    @staticmethod
    def send_message(message: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram error: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def format_data_message(all_data: List[Dict]) -> str:
        """Format exchange data into readable Telegram message"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        message = f"<b>🔄 Multi-Exchange Data Update</b>\n"
        message += f"<b>⏰ Time:</b> {timestamp}\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Process each exchange
        for data in all_data:
            exchange = data.get('exchange', 'Unknown')
            
            if 'error' in data:
                message += f"<b>❌ {exchange}</b>\n"
                message += f"Error: {data['error']}\n\n"
                continue
            
            message += f"<b>📊 {exchange}</b>\n"
            
            # Binance
            if exchange == "Binance":
                message += f"💵 Spot Price: ${data.get('spot_price', 0):,.2f}\n"
                message += f"📈 24h Volume: ${data.get('spot_volume_24h', 0)/1e9:.2f}B\n"
                message += f"🔓 Futures OI: {data.get('futures_oi', 0):,.0f} BTC\n"
                message += f"💰 Funding: {data.get('funding_rate', 0)*100:.4f}%\n"
            
            # OKX
            elif exchange == "OKX":
                message += f"💵 Spot Price: ${data.get('spot_price', 0):,.2f}\n"
                message += f"📈 24h Volume: ${data.get('spot_volume_24h', 0)/1e6:.2f}M\n"
                message += f"🔓 SWAP OI: {data.get('swap_oi', 0):,.0f}\n"
                message += f"💰 Funding: {data.get('funding_rate', 0)*100:.4f}%\n"
            
            # Bybit
            elif exchange == "Bybit":
                message += f"💵 Spot Price: ${data.get('spot_price', 0):,.2f}\n"
                message += f"📈 24h Volume: ${data.get('spot_volume_24h', 0)/1e9:.2f}B\n"
                message += f"🔓 Perpetual OI: {data.get('perpetual_oi', 0):,.0f}\n"
                message += f"💰 Funding: {data.get('funding_rate', 0)*100:.4f}%\n"
            
            # Deribit
            elif exchange == "Deribit":
                message += f"💵 Spot Price: ${data.get('spot_price', 0):,.2f}\n"
                message += f"📞 Call OI: {data.get('total_call_oi', 0):,.0f} BTC\n"
                message += f"📉 Put OI: {data.get('total_put_oi', 0):,.0f} BTC\n"
                message += f"⚖️ PCR Ratio: {data.get('pcr_ratio', 0):.2f}\n"
            
            # Delta Exchange
            elif exchange == "Delta Exchange":
                message += f"💵 Spot Price: ${data.get('spot_price', 0):,.2f}\n"
                message += f"📈 24h Volume: ${data.get('volume_24h', 0)/1e6:.2f}M\n"
                message += f"🔓 Open Interest: {data.get('oi', 0):,.0f}\n"
            
            message += "\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "<i>Next update in 5 minutes</i>"
        
        return message


# =============================================================================
# MAIN DATA FETCHING FUNCTION
# =============================================================================

def fetch_and_send_data():
    """Main function to fetch data from all exchanges and send to Telegram"""
    logger.info("🚀 Starting data fetch cycle...")
    
    all_data = []
    
    # Fetch from all exchanges
    logger.info("📡 Fetching Binance data...")
    all_data.append(BinanceFetcher.get_btc_data())
    
    logger.info("📡 Fetching OKX data...")
    all_data.append(OKXFetcher.get_btc_data())
    
    logger.info("📡 Fetching Bybit data...")
    all_data.append(BybitFetcher.get_btc_data())
    
    logger.info("📡 Fetching Deribit data...")
    all_data.append(DeribitFetcher.get_btc_data())
    
    logger.info("📡 Fetching Delta Exchange data...")
    all_data.append(DeltaExchangeFetcher.get_btc_data())
    
    # Format and send to Telegram
    logger.info("📤 Sending data to Telegram...")
    message = TelegramSender.format_data_message(all_data)
    TelegramSender.send_message(message)
    
    logger.info("✅ Data fetch cycle completed\n")


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler():
    """Run the scheduler to fetch data every 5 minutes"""
    logger.info("="*60)
    logger.info("🤖 Multi-Exchange Data Fetcher - Testing Mode")
    logger.info("="*60)
    logger.info(f"📅 Started at: {datetime.now(timezone.utc)}")
    logger.info(f"⏱️  Fetch Interval: Every 5 minutes")
    logger.info(f"📱 Telegram Chat ID: {TELEGRAM_CHAT_ID}")
    logger.info("="*60 + "\n")
    
    # Run immediately on start
    fetch_and_send_data()
    
    # Schedule every 5 minutes
    schedule.every(5).minutes.do(fetch_and_send_data)
    
    logger.info("⏳ Scheduler started. Press Ctrl+C to stop.\n")
    
    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every 30 seconds


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Validate environment variables
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file")
        exit(1)
    
    if not DELTA_API_KEY or not DELTA_API_SECRET:
        logger.warning("⚠️  Delta Exchange API credentials missing. Delta data will fail.")
    
    try:
        run_scheduler()
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
