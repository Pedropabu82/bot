import ccxt.async_support as ccxt
import json
import logging

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, testnet=True):
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_key']
        api_secret = config['api_secret']

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.exchange.set_sandbox_mode(testnet)
        logger.info(f"BinanceClient initialized ({'testnet' if testnet else 'mainnet'})")

    async def close(self):
        await self.exchange.close()
        logger.info("BinanceClient closed")
