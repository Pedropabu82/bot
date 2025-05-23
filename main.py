import asyncio, json, logging
from api_client import BinanceClient
from live_strategy import LiveMAStrategy

logging.basicConfig(level=logging.INFO)

async def main():
    with open("config.json","r") as f:
        cfg = json.load(f)

    client = BinanceClient(testnet=cfg.get("testnet",True))
    strategy = LiveMAStrategy(client, cfg.get("symbols",[]), cfg.get("timeframes",[]), config=cfg)
    await strategy.async_init()
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping...")
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
