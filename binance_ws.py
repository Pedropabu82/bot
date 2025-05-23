import asyncio
import json
import websockets
import logging
import pandas as pd
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

async def fetch_historical_klines(exchange, symbol, timeframe, limit=100):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Fetched historical klines for {symbol} {timeframe}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch historical klines for {symbol} {timeframe}: {e}")
        return None

async def start_streams(symbols, timeframes, strategy, valid_timeframes, config):
    uris = [
        "wss://stream.binancefuture.com/stream",
        "wss://fstream.binance.com/stream"
    ]
    streams = [
        f"{symbol.lower()}@kline_{tf}" for symbol in symbols
        for tf in (config.get('ws_timeframes', timeframes))
    ] + [
        f"{symbol.lower()}@ticker" for symbol in symbols
    ]
    stream_param = "/".join(streams)
    current_uri_index = 0
    # Use mainnet endpoint first if not using testnet
    if not config.get('testnet', True):
        current_uri_index = 1

    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.set_sandbox_mode(True)

    reconnect_delay = 5
    max_reconnect_delay = 120
    reconnect_attempts = 0
    max_attempts = 3

    while True:
        full_url = f"{uris[current_uri_index]}?streams={stream_param}"
        if reconnect_attempts >= max_attempts:
            logger.info(f"Max WebSocket attempts ({max_attempts}) reached, switching to REST API fallback")
            for symbol in symbols:
                for tf in timeframes:
                    df = await fetch_historical_klines(exchange, symbol, tf)
                    if df is not None:
                        await strategy.process_timeframe_data(symbol, tf, df)
                    else:
                        logger.warning(f"REST API failed for {symbol} {tf}, skipping data update")
            await asyncio.sleep(60)
            reconnect_attempts = 0
            reconnect_delay = 5
            continue

        try:
            async with websockets.connect(full_url, ping_interval=30, ping_timeout=10) as ws:
                logger.info(f"WebSocket connected to {full_url}: {stream_param}")
                reconnect_attempts = 0
                reconnect_delay = 5
                while True:
                    response = await ws.recv()
                    message = json.loads(response)
                    data = message.get('data', {})
                    stream_name = message.get('stream', '')
                    if not data or not stream_name:
                        continue

                    symbol = data.get('s', '').upper()
                    if not symbol:
                        continue

                    if '@kline_' in stream_name:
                        kline = data.get('k', {})
                        if not kline:
                            continue
                        # Only process if the kline/candle is closed (ignore interim updates)
                        if not kline.get('x', False):
                            continue
                        timeframe = stream_name.split('@kline_')[1]
                        if timeframe not in valid_timeframes:
                            logger.warning(f"Received invalid timeframe {timeframe} for {symbol}")
                            continue
                        df = pd.DataFrame({
                            'timestamp': [pd.Timestamp(kline['t'], unit='ms')],
                            'open': [float(kline['o'])],
                            'high': [float(kline['h'])],
                            'low': [float(kline['l'])],
                            'close': [float(kline['c'])],
                            'volume': [float(kline['v'])]
                        })
                        logger.info(f"Received WebSocket kline for {symbol} {timeframe}: timestamp={df['timestamp'].iloc[0]}, close={df['close'].iloc[0]}")
                        await strategy.process_timeframe_data(symbol, timeframe, df)

                    elif '@ticker' in stream_name:
                        price = float(data.get('c', 0))
                        if price > 0:
                            await strategy.process_tick(symbol, price)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket closed on {uris[current_uri_index]}: {e}")
            # Switch to alternate URI on connection issues
            current_uri_index = (current_uri_index + 1) % len(uris)
        except asyncio.CancelledError:
            logger.info("WebSocket stream task cancelled")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        reconnect_attempts += 1
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    await exchange.close()
