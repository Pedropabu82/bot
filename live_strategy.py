import pandas as pd
import numpy as np
import talib
import logging
import asyncio
import datetime
from rich.console import Console
from rich.table import Table

from optimize_indicators import IndicatorOptimizer
logger = logging.getLogger(__name__)

class LiveMAStrategy:
    def __init__(self, client, symbols, timeframes, config=None):
        self.client = client
        self.symbols = symbols
        self.timeframes = timeframes
        self.config = config or {}
        self.leverage = {symbol: self.config.get('leverage', 20) for symbol in symbols}  # Per-symbol leverage
        self.tp_roi = self.config.get('tp_roi', 1.5)   # ROI targets (%)
        self.sl_roi = self.config.get('sl_roi', 5.0)
        self.break_even_threshold = 1.5  # 1.5% price move for break-even adjustment
        self.trailing_activation_pct = 2.0  # 2% price move for trailing stop activation
        self.trailing_callback_rate = 0.3   # 0.3% trailing stop callback rate
        self.min_score = self.config.get('min_score', 3)
        self.cooldown_seconds = self.config.get('cooldown_pair_sec', self.config.get('cooldown_seconds', 900))
        self.price_offset_pct = self.config.get('price_offset_pct', 0.001)
        self.retry_offset_pct = self.config.get('retry_offset_pct', 0.001)
        self.entry_timeout_sec = self.config.get('entry_timeout_sec', 15)
        self.signal_priority = self.config.get('signal_priority', True)
        self.maintenance_margin_rate = 0.005
        self.console = Console()
        self.in_position = {symbol: False for symbol in symbols}
        self.position_side = {symbol: None for symbol in symbols}
        self.entry_price = {symbol: 0.0 for symbol in symbols}
        self.sl_price = {symbol: 0.0 for symbol in symbols}
        self.tp_price = {symbol: 0.0 for symbol in symbols}
        self.quantity = {symbol: 0.0 for symbol in symbols}
        self.latest_close = {symbol: 0.0 for symbol in symbols}
        self.data = {symbol: {tf: pd.DataFrame() for tf in timeframes} for symbol in symbols}
        self.last_signal = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.unrealized_pnl = {symbol: 0.0 for symbol in symbols}
        self.realized_pnl = {symbol: 0.0 for symbol in symbols}
        self.funding_fee = {symbol: 0.0 for symbol in symbols}
        self.commission = {symbol: 0.0 for symbol in symbols}
        self.margin_used = {symbol: 0.0 for symbol in symbols}
        self.liquidation_price = {symbol: 0.0 for symbol in symbols}
        self.margin_ratio = {symbol: 0.0 for symbol in symbols}
        self.position_timeframe = {symbol: None for symbol in symbols}
        self.update_positions_task = None
        self.monitor_positions_task = None
        self.performance_window = 30
        self.min_win_rate = 0.45
        self.disable_timeframes = {symbol: set() for symbol in symbols}
        self.last_adaptation = datetime.datetime.utcnow()
        self.debug_signals = True
        self.force_entry = False
        self.last_closed_time = {symbol: None for symbol in symbols}
        self.tp_order_id = {symbol: None for symbol in symbols}
        self.sl_order_id = {symbol: None for symbol in symbols}
        self.trailing_order_id = {symbol: None for symbol in symbols}
        self.long_timeframes = ["1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        self.short_timeframes = ["5m", "15m", "30m"]
        self.price_precision = {symbol: 2 for symbol in symbols}      # Default (will be updated in async_init)
        self.quantity_precision = {symbol: 3 for symbol in symbols}   # Default (will be updated in async_init)

    def get_cooldown_for_timeframe(self, symbol, timeframe):
        tf_seconds = {
            '5m': 60, '15m': 120, '30m': 180, '1h': 300, '2h': 600,
            '4h': 900, '6h': 1200, '8h': 1800, '12h': 2400, '1d': 3600
        }
        df = self.data[symbol].get('5m', pd.DataFrame())
        if len(df) >= 20:
            volatility = df['close'].tail(20).pct_change().std() * 100
            cooldown = min(self.cooldown_seconds * (1 + volatility), 1200)
        else:
            cooldown = self.cooldown_seconds
        return min(cooldown, tf_seconds.get(timeframe, self.cooldown_seconds))

    def check_cooldown(self, symbol, timeframe=None):
        if self.last_closed_time.get(symbol) is not None:
            cooldown = self.get_cooldown_for_timeframe(symbol, timeframe) if timeframe else self.cooldown_seconds
            elapsed = (datetime.datetime.utcnow() - self.last_closed_time[symbol]).total_seconds()
            if elapsed < cooldown:
                if self.debug_signals:
                    logger.info(f"Cooldown active for {symbol} ({timeframe or 'default'}): waiting {cooldown - elapsed:.0f} seconds")
                return False
        return True

    def update_close_time(self, symbol):
        self.last_closed_time[symbol] = datetime.datetime.utcnow()

    async def async_init(self):
        # Fetch exchange info for symbol-specific precision and leverage
        try:
            exchange_info = await self.client.exchange.fapiPublicGetExchangeInfo()
            for market in exchange_info['symbols']:
                symbol = next((s for s in self.symbols if s.replace('/', '') == market['symbol']), None)
                if symbol:
                    try:
                        price_precision = int(market['pricePrecision'])
                        quantity_precision = int(market['quantityPrecision'])
                    except (TypeError, ValueError) as e:
                        logger.error(f"Invalid precision for {symbol}: pricePrecision={market['pricePrecision']}, quantityPrecision={market['quantityPrecision']}, error={e}")
                        price_precision = 2
                        quantity_precision = 3
                    self.price_precision[symbol] = price_precision
                    self.quantity_precision[symbol] = quantity_precision
                    logger.info(f"Set precision for {symbol}: price={self.price_precision[symbol]}, quantity={self.quantity_precision[symbol]}")
        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")

        # Set leverage for each symbol
        for symbol in self.symbols:
            try:
                symbol_clean = symbol.replace('/', '')
                response = await self.client.exchange.fapiPrivatePostLeverage({
                    'symbol': symbol_clean, 'leverage': self.leverage[symbol]
                })
                self.leverage[symbol] = int(response['leverage'])
                logger.info(f"Set leverage {self.leverage[symbol]}x for {symbol}: {response}")
            except Exception as e:
                logger.error(f"Failed to set leverage for {symbol}: {e}")
        await self.fetch_initial_history()
        # Initialize per-symbol locks to prevent concurrent order placement
        self.entry_locks = {symbol: asyncio.Lock() for symbol in self.symbols}
        self.update_positions_task = asyncio.create_task(self.update_positions_loop())
        self.monitor_positions_task = asyncio.create_task(self.monitor_positions_loop())

    async def fetch_initial_history(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                limit = {
                    '5m': 300, '15m': 200, '30m': 150, '1h': 120, '2h': 100,
                    '4h': 100, '6h': 80, '8h': 100, '12h': 80, '1d': 60
                }.get(tf, 100)
                max_attempts = 3
                attempt = 1
                while attempt <= max_attempts:
                    try:
                        ohlcv = await self.client.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.data[symbol][tf] = df
                        if not df.empty:
                            self.latest_close[symbol] = df['close'].iloc[-1]
                            logger.info(f"Set initial latest_close for {symbol} to {self.latest_close[symbol]} from {tf}")
                        logger.info(f"Fetched {len(df)} candles for {symbol} {tf}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to fetch history for {symbol} {tf} (attempt {attempt}): {e}")
                        attempt += 1
                        if attempt <= max_attempts:
                            await asyncio.sleep(1)
                        continue
                if attempt > max_attempts:
                    logger.error(f"Failed to fetch history for {symbol} {tf} after {max_attempts} attempts")

    async def update_positions_loop(self):
        while True:
            try:
                await self.fetch_positions()
                for symbol in self.symbols:
                    if self.in_position[symbol]:
                        async with self.entry_locks[symbol]:
                            await self.manage_existing_position(symbol)
            except Exception as e:
                logger.error(f"Failed to update positions: {e}")
            await asyncio.sleep(3)

    async def monitor_positions_loop(self):
        while True:
            try:
                for symbol in self.symbols:
                    await self.evaluate_strategy(symbol)
                    logger.debug(f"Periodic evaluation triggered for {symbol}")
            except Exception as e:
                logger.error(f"Failed to monitor positions: {e}")
            await asyncio.sleep(5)

    async def fetch_positions(self):
        try:
            positions = await self.client.exchange.fapiPrivateV2GetPositionRisk()
            for pos in positions:
                symbol = pos.get('symbol')
                symbol_with_slash = next((s for s in self.symbols if s.replace('/', '') == symbol), None)
                if symbol_with_slash not in self.symbols:
                    continue
                qty = float(pos.get('positionAmt', 0) or 0)
                entry = float(pos.get('entryPrice', 0) or 0)
                mark = float(pos.get('markPrice', entry) or entry)
                leverage = float(pos.get('leverage', self.leverage[symbol_with_slash]) or self.leverage[symbol_with_slash])
                notional = abs(qty * mark)
                margin = notional / leverage if leverage else 0.0
                liq_price = float(pos.get('liquidationPrice', 0) or 0)

                # Calculate PNL for current position
                if qty > 0:  # Long
                    un_pnl = (mark - entry) * qty * 1
                elif qty < 0:  # Short
                    un_pnl = (entry - mark) * qty * 1
                else:
                    un_pnl = 0.0
                margin_ratio = abs(un_pnl) / margin * 100 if margin else 0.0

                self.in_position[symbol_with_slash] = abs(qty) > 0
                self.position_side[symbol_with_slash] = "long" if qty > 0 else ("short" if qty < 0 else None)
                self.leverage[symbol_with_slash] = leverage
                if self.in_position[symbol_with_slash] is False and self.position_side[symbol_with_slash] is not None:
                    logger.warning(f"Detected manual close of position for {symbol_with_slash}. Resetting state.")
                    self.position_side[symbol_with_slash] = None
                    self.entry_price[symbol_with_slash] = 0.0
                    self.sl_price[symbol_with_slash] = 0.0
                    self.tp_price[symbol_with_slash] = 0.0
                    self.quantity[symbol_with_slash] = 0.0
                    self.position_timeframe[symbol_with_slash] = None
                    self.tp_order_id[symbol_with_slash] = None
                    self.sl_order_id[symbol_with_slash] = None
                    self.trailing_order_id[symbol_with_slash] = None

                self.entry_price[symbol_with_slash] = entry if abs(qty) > 0 else 0.0
                self.quantity[symbol_with_slash] = abs(qty)
                self.latest_close[symbol_with_slash] = mark
                self.unrealized_pnl[symbol_with_slash] = un_pnl
                self.margin_used[symbol_with_slash] = margin
                self.liquidation_price[symbol_with_slash] = liq_price
                self.margin_ratio[symbol_with_slash] = margin_ratio
                roi_pct = (un_pnl / margin * 100) if margin != 0 else 0.0
                logger.info(f"PNL for {symbol_with_slash}: {un_pnl:.2f} USDT, ROI={roi_pct:.2f}%")

                funding = await self.fetch_funding_fee(symbol_with_slash)
                self.funding_fee[symbol_with_slash] = funding
                commission = await self.fetch_commission(symbol_with_slash)
                self.commission[symbol_with_slash] = commission
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    async def fetch_funding_fee(self, symbol):
        try:
            symbol_clean = symbol.replace('/', '')
            rows = await self.client.exchange.fapiPrivateGetIncome(
                {"symbol": symbol_clean, "incomeType": "FUNDING_FEE", "limit": 10}
            )
            funding = sum(float(x['income']) for x in rows)
            return funding
        except Exception as e:
            logger.warning(f"Failed to fetch funding for {symbol}: {e}")
            return 0.0

    async def fetch_commission(self, symbol):
        try:
            symbol_clean = symbol.replace('/', '')
            trades = await self.client.exchange.fapiPrivateGetUserTrades(
                {"symbol": symbol_clean, "limit": 10}
            )
            commission = sum(float(t['commission']) for t in trades)
            return commission
        except Exception as e:
            logger.warning(f"Failed to fetch commission for {symbol}: {e}")
            return 0.0

    async def process_timeframe_data(self, symbol, timeframe, df):
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if timeframe not in self.timeframes:
            logger.warning(f"Received invalid timeframe {timeframe} for {symbol}")
            return
        if df is None or df.empty:
            logger.warning(f"Empty OHLCV data for {symbol} {timeframe}")
            return

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in OHLCV data for {symbol} {timeframe}")
            return

        try:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            self.data[symbol][timeframe] = pd.concat([self.data[symbol][timeframe], df]) \
                                             .drop_duplicates(subset=['timestamp'], keep='last') \
                                             .tail(100)
            self.latest_close[symbol] = df['close'].iloc[-1]
            logger.info(f"Processed WebSocket data for {symbol} {timeframe}: timestamp={df['timestamp'].iloc[-1]}, close={self.latest_close[symbol]}")
            await self.evaluate_strategy(symbol)
            if not self.signal_priority:
                logger.info(f"Evaluated strategy for {symbol} {timeframe}: signals updated")
        except Exception as e:
            logger.warning(f"Invalid OHLCV data format for {symbol} {timeframe}: {e}")

    async def process_tick(self, symbol, price):
        if price > 0:
            self.latest_close[symbol] = price
            logger.debug(f"Updated latest_close for {symbol} to {price} via tick")

    def calculate_tp_sl(self, symbol):
        entry = self.entry_price[symbol]
        side = self.position_side[symbol]
        leverage = self.leverage[symbol]
        logger.debug(f"Calculating TP/SL for {symbol}: entry={entry:.2f}, side={side}, leverage={leverage}x")

        # Adjust TP/SL based on ROI targets (tp_roi, sl_roi are in percent)
        if side == "long":
            tp_price = entry * (1 + self.tp_roi / (100 * leverage))
            sl_price = entry * (1 - self.sl_roi / (100 * leverage))
        else:  # short
            tp_price = entry * (1 - self.tp_roi / (100 * leverage))
            sl_price = entry * (1 + self.sl_roi / (100 * leverage))

        # Round to symbol-specific price precision
        tp_price = round(tp_price, self.price_precision[symbol])
        sl_price = round(sl_price, self.price_precision[symbol])

        # Validate TP/SL levels relative to entry
        if side == "long" and (tp_price <= entry or sl_price >= entry):
            logger.error(f"Invalid TP/SL for {symbol} long: TP={tp_price:.2f}, SL={sl_price:.2f}, entry={entry:.2f}")
        elif side == "short" and (tp_price >= entry or sl_price <= entry):
            logger.error(f"Invalid TP/SL for {symbol} short: TP={tp_price:.2f}, SL={sl_price:.2f}, entry={entry:.2f}")

        logger.info(f"Calculated TP/SL for {symbol} ({side}): TP={tp_price:.2f}, SL={sl_price:.2f}, entry={entry:.2f}, leverage={leverage}x")
        return tp_price, sl_price

    def get_signal_for_timeframe(self, symbol, timeframe):
        df = self.data[symbol][timeframe]
        if len(df) < 60:
            logger.debug(f"Not enough data for {symbol} {timeframe}: {len(df)} candles")
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicator_config = self.config.get('indicators', {}).get(symbol, {})
        ema_short = indicator_config.get('ema_short', 12)
        ema_long = indicator_config.get('ema_long', 26)
        rsi_period = indicator_config.get('rsi_period', 14)
        stoch_rsi_period = indicator_config.get('stoch_rsi_period', 14)
        stoch_rsi_fastk = indicator_config.get('stoch_rsi_fastk', 3)
        stoch_rsi_fastd = indicator_config.get('stoch_rsi_fastd', 3)
        adx_period = indicator_config.get('adx_period', 14)
        macd_fast = indicator_config.get('macd_fast', 12)
        macd_slow = indicator_config.get('macd_slow', 26)
        macd_signal = indicator_config.get('macd_signal', 9)
        tema_period = indicator_config.get('tema_period', 21)

        try:
            stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(
                close, timeperiod=stoch_rsi_period,
                fastk_period=stoch_rsi_fastk,
                fastd_period=stoch_rsi_fastd, fastd_matype=0
            )
        except Exception:
            stoch_rsi_k = pd.Series(np.nan, index=close.index)
            stoch_rsi_d = pd.Series(np.nan, index=close.index)
        stoch_cross_long = False
        stoch_cross_short = False
        if len(stoch_rsi_k) > 2 and not np.isnan(stoch_rsi_k.iloc[-2]) and not np.isnan(stoch_rsi_d.iloc[-2]):
            if stoch_rsi_k.iloc[-2] < stoch_rsi_d.iloc[-2] and stoch_rsi_k.iloc[-1] > stoch_rsi_d.iloc[-1]:
                stoch_cross_long = True
            if stoch_rsi_k.iloc[-2] > stoch_rsi_d.iloc[-2] and stoch_rsi_k.iloc[-1] < stoch_rsi_d.iloc[-1]:
                stoch_cross_short = True

        ema9 = talib.EMA(close, timeperiod=ema_short)
        ema21 = talib.EMA(close, timeperiod=ema_long)
        ema13 = talib.EMA(close, timeperiod=13)
        ema55 = talib.EMA(close, timeperiod=55)
        ema9_21_cross_long = ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]
        ema9_21_cross_short = ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]
        ema13_55_cross_long = ema13.iloc[-2] < ema55.iloc[-2] and ema13.iloc[-1] > ema55.iloc[-1]
        ema13_55_cross_short = ema13.iloc[-2] > ema55.iloc[-2] and ema13.iloc[-1] < ema55.iloc[-1]
        ema_cross_confirm_long = ema9_21_cross_long and ema13_55_cross_long
        ema_cross_confirm_short = ema9_21_cross_short and ema13_55_cross_short

        window = 20
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-9)
        high_vol = vol_z.iloc[-1] > 2
        low_vol = vol_z.iloc[-1] < -2

        adx = talib.ADX(high, low, close, timeperiod=adx_period)
        tf_adx_thresholds = {
            "5m": 10, "15m": 15, "30m": 18, "1h": 20, "2h": 20,
            "4h": 20, "6h": 18, "8h": 30, "12h": 35, "1d": 40
        }
        adx_thresh = tf_adx_thresholds.get(timeframe, 20)
        adx_strong = adx.iloc[-1] > adx_thresh

        rsi = talib.RSI(close, timeperiod=rsi_period)
        rsi_pct_high = np.percentile(rsi.dropna()[-30:], 70)
        rsi_pct_low = np.percentile(rsi.dropna()[-30:], 30)
        rsi_long = rsi.iloc[-1] < rsi_pct_low and 30 < rsi.iloc[-1] < 70
        rsi_short = rsi.iloc[-1] > rsi_pct_high and 30 < rsi.iloc[-1] < 70

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        macdhist_slope_pos = macdhist.iloc[-1] > 0 and (macdhist.iloc[-1] - macdhist.iloc[-2]) > 0
        macdhist_slope_neg = macdhist.iloc[-1] < 0 and (macdhist.iloc[-1] - macdhist.iloc[-2]) < 0

        try:
            tema = talib.TEMA(close, timeperiod=tema_period)
            tema_slope = tema.iloc[-1] - tema.iloc[-5]
        except Exception:
            tema = pd.Series(np.nan, index=close.index)
            tema_slope = 0
        tema_up = tema_slope > 0
        tema_down = tema_slope < 0

        # If trend is weak, skip signals for this timeframe
        if adx.iloc[-1] <= adx_thresh:
            logger.debug(f"Weak trend for {symbol} {timeframe}: ADX={adx.iloc[-1]:.2f}")
            return None

        # Additional indicators: ATR and OBV
        atr = talib.ATR(high, low, close, timeperiod=14)
        atr_volatility = (high.iloc[-1] - low.iloc[-1]) < 1.5 * atr.iloc[-1]

        obv = talib.OBV(close, volume)
        obv_slope = obv.iloc[-1] - obv.iloc[-5]
        obv_up = obv_slope > 0
        obv_down = obv_slope < 0

        # Scoring for long/short conditions
        indicator_long = sum([
            2 * ema_cross_confirm_long,
            1.5 * (adx_strong and tema_up),
            1 * stoch_cross_long,
            1 * high_vol,
            1 * rsi_long,
            1 * macdhist_slope_pos,
            1 * tema_up,
            1 * atr_volatility,
            1 * obv_up
        ])
        indicator_short = sum([
            2 * ema_cross_confirm_short,
            1.5 * (adx_strong and tema_down),
            1 * stoch_cross_short,
            1 * low_vol,
            1 * rsi_short,
            1 * macdhist_slope_neg,
            1 * tema_down,
            1 * atr_volatility,
            1 * obv_down
        ])

        signal = None
        if not self.check_cooldown(symbol, timeframe):
            logger.debug(f"Cooldown active for {symbol} {timeframe}: skipping entry signals")
            return None

        if indicator_long >= self.min_score:
            logger.info(f"Buy conditions met for {symbol} {timeframe} with score {indicator_long}")
            signal = "long"
        elif indicator_short >= self.min_score:
            logger.info(f"Sell conditions met for {symbol} {timeframe} with score {indicator_short}")
            signal = "short"

        if self.force_entry:
            signal = "long"

        logger.debug(
            f"Signal for {symbol} {timeframe}: {signal} "
            f"(StochRSI(L:{stoch_cross_long},S:{stoch_cross_short}), "
            f"EMA Cross(L:{ema_cross_confirm_long},S:{ema_cross_confirm_short}), "
            f"VolZ(L:{high_vol},S:{low_vol}), "
            f"ADX(L:{adx_strong and tema_up},S:{adx_strong and tema_down}), "
            f"RSI(L:{rsi_long},S:{rsi_short}), "
            f"MACD(L:{macdhist_slope_pos},S:{macdhist_slope_neg}), "
            f"TEMA(L:{tema_up},S:{tema_down}), "
            f"Long count: {indicator_long}, Short count: {indicator_short})"
        )
        return signal

    def check_multi_timeframe_signal(self, symbol):
        long_timeframes = [tf for tf in self.long_timeframes if tf in self.timeframes]
        short_timeframes = [tf for tf in self.short_timeframes if tf in self.timeframes]

        main_signal = None
        main_timeframe = None
        signals = {}

        for tf in long_timeframes:
            sig = self.get_signal_for_timeframe(symbol, tf)
            signals[tf] = sig
            logger.debug(f"Main signal check for {symbol} {tf}: {sig}")
            if sig is not None:
                main_signal = sig
                main_timeframe = tf
                break

        if not main_signal:
            logger.debug(f"No main signal for {symbol} from long timeframes")
            return None, None

        confirm = False
        conflicting_signals = []
        for tf in short_timeframes:
            sig = self.get_signal_for_timeframe(symbol, tf)
            signals[tf] = sig
            logger.debug(f"Confirmation signal check for {symbol} {tf}: {sig}")
            if sig == main_signal:
                confirm = True
            elif sig is not None and sig != main_signal:
                conflicting_signals.append((tf, sig))

        if conflicting_signals:
            logger.warning(
                f"Signal conflict for {symbol}: main signal {main_signal} on {main_timeframe}, "
                f"conflicting signals: {', '.join([f'{tf}: {sig}' for tf, sig in conflicting_signals])}"
            )

        if confirm:
            logger.debug(f"Confirmed {main_signal} signal for {symbol} on {main_timeframe}")
            return main_signal, main_timeframe
        logger.debug(f"No confirmation for {main_signal} signal for {symbol}")
        return None, None

    async def manage_existing_position(self, symbol):
        if not self.in_position[symbol]:
            return
        try:
            open_orders = await self.client.exchange.fetch_open_orders(symbol)
            side = self.position_side[symbol]
            qty = self.quantity[symbol]
            expected_side = 'sell' if side == 'long' else 'buy'

            # Verify existing TP/SL/Trailing orders
            tp_exists = any(
                order['id'] == self.tp_order_id[symbol] and 
                order['type'].upper() == 'TAKE_PROFIT_MARKET' and 
                abs(float(order['stopPrice']) - self.tp_price[symbol]) < 0.01 * self.tp_price[symbol] and 
                order.get('reduceOnly', False) and 
                order['side'].lower() == expected_side
                for order in open_orders
            )
            sl_exists = any(
                order['id'] == self.sl_order_id[symbol] and 
                order['type'].upper() == 'STOP_MARKET' and 
                abs(float(order['stopPrice']) - self.sl_price[symbol]) < 0.01 * self.sl_price[symbol] and 
                order.get('reduceOnly', False) and 
                order['side'].lower() == expected_side
                for order in open_orders
            )
            trailing_exists = any(
                order['id'] == self.trailing_order_id[symbol] and 
                order['type'].upper() == 'TRAILING_STOP_MARKET' and 
                order.get('reduceOnly', False) and 
                order['side'].lower() == expected_side
                for order in open_orders
            )

            # Cancel any incorrect or outdated orders
            for order in open_orders:
                is_invalid = (
                    (order['type'].upper() in ['LIMIT', 'STOP', 'TAKE_PROFIT'] or
                     (order['type'].upper() == 'TAKE_PROFIT_MARKET' and 
                      abs(float(order['stopPrice']) - self.tp_price[symbol]) > 0.01 * self.tp_price[symbol]) or
                     (order['type'].upper() == 'STOP_MARKET' and 
                      abs(float(order['stopPrice']) - self.sl_price[symbol]) > 0.01 * self.sl_price[symbol]) or
                     (order['type'].upper() == 'TRAILING_STOP_MARKET' and 
                      order['id'] != self.trailing_order_id[symbol])) or
                    not order.get('reduceOnly', False) or
                    order['side'].lower() != expected_side
                )
                if is_invalid:
                    await self.client.exchange.cancel_order(order['id'], symbol)
                    logger.info(f"Canceled incorrect order for {symbol}: order_id={order['id']}, type={order['type']}, stopPrice={order.get('stopPrice', 'N/A')}, side={order['side']}, reduceOnly={order.get('reduceOnly', False)}")

            # Recalculate TP/SL if any expected order is missing
            if not tp_exists or not sl_exists or not trailing_exists:
                logger.warning(f"Missing orders for {symbol}: TP={tp_exists}, SL={sl_exists}, Trailing={trailing_exists}")
                self.tp_price[symbol], self.sl_price[symbol] = self.calculate_tp_sl(symbol)

            # Ensure Take-Profit order exists
            if not tp_exists:
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    try:
                        tp_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='TAKE_PROFIT_MARKET',
                            side=expected_side,
                            amount=qty,
                            params={
                                "stopPrice": self.tp_price[symbol],
                                "reduceOnly": True
                            }
                        )
                        self.tp_order_id[symbol] = tp_order['id']
                        logger.info(f"Placed TAKE_PROFIT_MARKET order for {symbol}: order_id={tp_order['id']}, stopPrice={self.tp_price[symbol]:.4f}, side={expected_side}, qty={qty}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to place TAKE_PROFIT_MARKET for {symbol} (attempt {attempt}): {e}")
                        if attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place TAKE_PROFIT_MARKET for {symbol} after {max_attempts} attempts")
                            return

            # Ensure Stop-Loss order exists
            if not sl_exists:
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    try:
                        sl_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='STOP_MARKET',
                            side=expected_side,
                            amount=qty,
                            params={
                                "stopPrice": self.sl_price[symbol],
                                "reduceOnly": True
                            }
                        )
                        self.sl_order_id[symbol] = sl_order['id']
                        logger.info(f"Placed STOP_MARKET order for {symbol}: order_id={sl_order['id']}, stopPrice={self.sl_price[symbol]:.4f}, side={expected_side}, qty={qty}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to place STOP_MARKET for {symbol} (attempt {attempt}): {e}")
                        if attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place STOP_MARKET for {symbol} after {max_attempts} attempts")
                            return

            # Ensure Trailing Stop order exists
            if not trailing_exists:
                max_attempts = 3
                activation_price = self.entry_price[symbol] * (1 + self.trailing_activation_pct / 100 if side == 'long' else 1 - self.trailing_activation_pct / 100)
                activation_price = round(activation_price, self.price_precision[symbol])
                for attempt in range(1, max_attempts + 1):
                    try:
                        trailing_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='TRAILING_STOP_MARKET',
                            side=expected_side,
                            amount=qty,
                            params={
                                "activationPrice": activation_price,
                                "callbackRate": self.trailing_callback_rate,
                                "reduceOnly": True
                            }
                        )
                        self.trailing_order_id[symbol] = trailing_order['id']
                        logger.info(f"Placed TRAILING_STOP_MARKET order for {symbol}: order_id={trailing_order['id']}, activationPrice={activation_price:.4f}, callbackRate={self.trailing_callback_rate}, qty={qty}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to place TRAILING_STOP_MARKET for {symbol} (attempt {attempt}): {e}")
                        if attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place TRAILING_STOP_MARKET for {symbol} after {max_attempts} attempts")
                            return

            # Break-even stop adjustment
            price = self.latest_close[symbol]
            break_even_trigger = self.entry_price[symbol] * (1 + self.break_even_threshold / 100 if side == "long" else 1 - self.break_even_threshold / 100)
            if (side == "long" and price >= break_even_trigger) or (side == "short" and price <= break_even_trigger):
                new_sl = self.entry_price[symbol]
                new_sl = round(new_sl, self.price_precision[symbol])
                if abs(new_sl - self.sl_price[symbol]) > 0.01 * self.sl_price[symbol]:
                    # Cancel existing SL to place break-even SL
                    if self.sl_order_id[symbol]:
                        try:
                            await self.client.exchange.cancel_order(self.sl_order_id[symbol], symbol)
                            logger.info(f"Canceled SL order for {symbol} to set break-even: order_id={self.sl_order_id[symbol]}")
                            self.sl_order_id[symbol] = None
                        except Exception as e:
                            logger.warning(f"Failed to cancel SL order for {symbol}: {e}")
                    # Place new SL at break-even price
                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        try:
                            sl_order = await self.client.exchange.create_order(
                                symbol=symbol,
                                type='STOP_MARKET',
                                side=expected_side,
                                amount=qty,
                                params={
                                    "stopPrice": new_sl,
                                    "reduceOnly": True
                                }
                            )
                            self.sl_order_id[symbol] = sl_order['id']
                            self.sl_price[symbol] = new_sl
                            logger.info(f"Set break-even STOP_MARKET for {symbol} ({side}): order_id={sl_order['id']}, stopPrice={new_sl:.4f}, qty={qty}")
                            break
                        except Exception as e:
                            logger.error(f"Failed to set break-even STOP_MARKET for {symbol} (attempt {attempt}): {e}")
                            if attempt < max_attempts:
                                await asyncio.sleep(1)
                            else:
                                logger.error(f"Failed to set break-even STOP_MARKET for {symbol} after {max_attempts} attempts")
                                return

        except Exception as e:
            logger.error(f"Failed to manage existing position for {symbol}: {e}")

    async def evaluate_strategy(self, symbol):
        async with self.entry_locks[symbol]:
            if not self.signal_priority:
                logger.debug(f"Evaluating strategy for {symbol}")
                self.automate_learning(symbol)

            # Initialize the indicator optimizer if not already done
            if not hasattr(self, 'indicator_optimizer'):
                self.indicator_optimizer = IndicatorOptimizer(self)

            signal, timeframe = self.check_multi_timeframe_signal(symbol)
            if signal and not self.in_position[symbol]:
                qty = await self.calc_order_qty(symbol, self.latest_close[symbol])
                logger.info(f"Placing {signal.upper()} order for {symbol} at price {self.latest_close[symbol]}")
                await self.open_position(symbol, signal, self.latest_close[symbol], qty, timeframe)
                self.position_timeframe[symbol] = timeframe
                logger.info(f"{datetime.datetime.now()} Entry on {symbol} (multi-TF): {signal.upper()}")
                return

            if self.in_position[symbol]:
                tf = self.position_timeframe[symbol]
                signal = self.get_signal_for_timeframe(symbol, tf) if tf else None
                side = self.position_side[symbol]
                price = self.latest_close[symbol]

                if not self.signal_priority:
                    logger.info(
                        f"Evaluating position for {symbol} ({side}, timeframe={tf}): "
                        f"current_price={price}, entry={self.entry_price[symbol]}, "
                        f"sl={self.sl_price[symbol]}, tp={self.tp_price[symbol]}, signal={signal}"
                    )

                should_close = False
                reason = ""
                if side == "long" and signal == "short":
                    opposite_signal_confirmed = False
                    for tf in self.short_timeframes:
                        if self.get_signal_for_timeframe(symbol, tf) == signal:
                            opposite_signal_confirmed = True
                            break
                    if opposite_signal_confirmed:
                        should_close = True
                        reason = "confirmed opposite signal"
                elif side == "short" and signal == "long":
                    opposite_signal_confirmed = False
                    for tf in self.short_timeframes:
                        if self.get_signal_for_timeframe(symbol, tf) == signal:
                            opposite_signal_confirmed = True
                            break
                    if opposite_signal_confirmed:
                        should_close = True
                        reason = "confirmed opposite signal"

                if should_close:
                    logger.info(f"Closing position for {symbol}: {reason}")
                    await self.close_position(symbol, price)
                elif not self.signal_priority:
                    logger.info(f"Position for {symbol} remains open: price={price}, sl={self.sl_price[symbol]}, tp={self.tp_price[symbol]}, signal={signal}")

                margin_ratio = self.margin_ratio.get(symbol, 0.0)
                if margin_ratio >= 80:
                    logger.warning(f"{symbol} margin ratio {margin_ratio:.2f}%: CLOSE TO LIQUIDATION!")

    async def calc_order_qty(self, symbol, price):
        if price <= 0:
            logger.error(f"Invalid price {price} for {symbol}: cannot calculate quantity")
            return 0.0
        try:
            balance = await self.client.exchange.fetch_balance()
            usdt_bal = balance['total'].get('USDT', 0.0)
        except Exception as e:
            logger.error(f"Failed to fetch balance for {symbol}: {e}")
            return 0.0
        if usdt_bal <= 0:
            logger.warning(f"No USDT balance available for {symbol}")
            return 0.0
        total_exposure = sum(self.margin_used.values())
        if usdt_bal > 0 and total_exposure / usdt_bal > 0.2:
            logger.warning(f"Exposure limit reached for {symbol}: {total_exposure/usdt_bal:.2%}")
            return 0.0
        # Calculate quantity based on risk percentage
        risk_pct = self.get_adaptive_risk(symbol)
        risk_amt = usdt_bal * risk_pct
        max_risk_amt = usdt_bal * 0.1
        risk_amt = min(risk_amt, max_risk_amt)
        qty = (risk_amt * self.leverage[symbol]) / price
        qty = round(qty, self.quantity_precision[symbol])
        min_qty = 10 ** (-self.quantity_precision[symbol])
        if qty < min_qty:
            logger.warning(f"Qty {qty} too low for {symbol}, forcing to min_qty={min_qty}")
            qty = min_qty
        logger.info(f"Calculated qty for {symbol}: {qty} (risk_amt={risk_amt:.2f}, price={price}, leverage={self.leverage[symbol]})")
        return qty

    def get_adaptive_risk(self, symbol):
        win_rate = self.get_win_rate(symbol)
        min_risk = 0.005
        max_risk = 0.02
        base_risk = 0.01
        if win_rate is None:
            return base_risk
        if win_rate < 0.5:
            return max(min_risk, base_risk * 0.5)
        elif win_rate > 0.65:
            return min(max_risk, base_risk * 1.5)
        return base_risk

    async def open_position(self, symbol, side, price, qty, timeframe):
        if qty <= 0:
            logger.warning(f"Qty zero, cannot open {side} for {symbol}")
            return
        if price <= 0:
            try:
                ticker = await self.client.exchange.fetch_ticker(symbol)
                price = ticker['last']
                logger.info(f"Fetched current price for {symbol}: {price}")
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")
                return

        max_attempts = 2
        attempt = 1

        while attempt <= max_attempts:
            try:
                order_side = 'buy' if side == "long" else 'sell'
                order_type = 'market' if self.config.get('mode') == 'taker' else self.config.get('order_types', {}).get(timeframe, 'limit')

                # Cancel any existing open orders for safety
                try:
                    open_orders = await self.client.exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        await self.client.exchange.cancel_order(order['id'], symbol)
                        logger.info(f"Canceled existing order for {symbol}: order_id={order['id']}, type={order['type']}, price={order.get('price', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Failed to cancel open orders for {symbol}: {e}")

                if order_type == "market":
                    # Place a market order for immediate execution
                    order = await self.client.exchange.create_market_order(symbol, order_side, qty)
                    logger.info(f"Placed MARKET {side.upper()} order for {symbol}: qty={qty}")
                else:
                    # Calculate dynamic limit price offset based on recent volatility
                    df = self.data[symbol].get('1m', self.data[symbol].get('5m', pd.DataFrame()))
                    if len(df) >= 10:
                        volatility = df['close'].tail(10).pct_change().std() * 100
                        offset_pct = min(max(volatility * 0.3, 0.01), 0.05)
                    else:
                        offset_pct = self.price_offset_pct

                    if attempt > 1:
                        # On retry, adjust the offset (e.g., tighter offset or use configured retry offset)
                        offset_pct = self.retry_offset_pct or offset_pct * 0.5
                        logger.info(f"Retry attempt {attempt} for {symbol}: adjusted offset_pct to {offset_pct}")

                    # Set limit price slightly better than current to act as maker
                    limit_price = price * (1 - offset_pct / 100) if side == "long" else price * (1 + offset_pct / 100)
                    try:
                        depth = await self.client.exchange.fetch_order_book(symbol, limit=20)
                        best_bid = depth['bids'][0][0] if depth['bids'] else price * 0.99
                        best_ask = depth['asks'][0][0] if depth['asks'] else price * 1.01
                        spread = best_ask - best_bid
                        spread_pct = (spread / best_ask) * 100
                        df_vol = self.data[symbol].get("1m", self.data[symbol].get("5m", pd.DataFrame()))
                        volatility = df_vol["close"].pct_change().tail(10).std() * 100 if len(df_vol) > 10 else 0.02
                        offset_pct = min(max(spread_pct + volatility, 0.01), 0.2)
                        logger.info(f"Dynamic offset_pct for {symbol}: spread={spread_pct:.4f}%, vol={volatility:.4f}%, final={offset_pct:.4f}%")
                    except Exception as e:
                        logger.warning(f"Failed to compute spread/volatility for {symbol}, using fallback offset_pct: {e}")
                        # fallback: use existing offset_pct

                    limit_price = round(limit_price, self.price_precision[symbol])
                    price_diff = abs(price - limit_price)
                    logger.info(
                        f"Placing MAKER {side.upper()} LIMIT order for {symbol} (attempt {attempt}): "
                        f"qty={qty}, limit_price={limit_price:.4f}, offset_pct={offset_pct}, price_diff={price_diff:.4f}"
                    )

                    order = await self.client.exchange.create_limit_order(symbol, order_side, qty, limit_price, params={"postOnly": True})

                    # Wait briefly for the limit order to fill, cancel if not filled in time
                    wait_time = self.entry_timeout_sec
                    interval = 0.2
                    checks = int(wait_time / interval)

                    for _ in range(checks):
                        await asyncio.sleep(interval)
                        order_status = await self.client.exchange.fetch_order(order['id'], symbol)
                        if order_status['status'] == 'closed':
                            logger.info(f"Order filled for {symbol}: order_id={order['id']}")
                            break
                    else:
                        await self.client.exchange.cancel_order(order['id'], symbol)
                        logger.info(
                            f"Canceled unfilled order for {symbol} (attempt {attempt}): "
                            f"order_id={order['id']}, limit_price={limit_price:.4f}"
                        )
                        attempt += 1
                        price = self.latest_close[symbol]
                        if price <= 0:
                            try:
                                ticker = await self.client.exchange.fetch_ticker(symbol)
                                price = ticker['last']
                                logger.info(f"Fetched current price for {symbol} retry: {price}")
                            except Exception as e:
                                logger.error(f"Failed to fetch retry price for {symbol}: {e}")
                                break
                        continue

                # If we reach here, position is considered open
                self.in_position[symbol] = True
                self.position_side[symbol] = side
                self.entry_price[symbol] = price
                self.quantity[symbol] = qty
                self.tp_price[symbol], self.sl_price[symbol] = self.calculate_tp_sl(symbol)
                self.log_trade(symbol, timeframe or "unknown", side, price, None, 0.0, "OPEN")

                # Place initial exit orders (TP, SL, Trailing) for the new position
                max_attempts = 3
                for tp_attempt in range(1, max_attempts + 1):
                    try:
                        tp_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='TAKE_PROFIT_MARKET',
                            side='sell' if side == 'long' else 'buy',
                            amount=qty,
                            params={
                                "stopPrice": self.tp_price[symbol],
                                "reduceOnly": True
                            }
                        )
                        self.tp_order_id[symbol] = tp_order['id']
                        logger.info(
                            f"Placed TAKE_PROFIT_MARKET order for {symbol}: order_id={tp_order['id']}, "
                            f"stopPrice={self.tp_price[symbol]:.4f}, side={'sell' if side == 'long' else 'buy'}, qty={qty}"
                        )
                        break
                    except Exception as e:
                        logger.error(f"Failed to place TAKE_PROFIT_MARKET for {symbol} (attempt {tp_attempt}): {e}")
                        if tp_attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place TAKE_PROFIT_MARKET for {symbol} after {max_attempts} attempts")
                            await self.close_position(symbol, price)
                            return

                for sl_attempt in range(1, max_attempts + 1):
                    try:
                        sl_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='STOP_MARKET',
                            side='sell' if side == 'long' else 'buy',
                            amount=qty,
                            params={
                                "stopPrice": self.sl_price[symbol],
                                "reduceOnly": True
                            }
                        )
                        self.sl_order_id[symbol] = sl_order['id']
                        logger.info(
                            f"Placed STOP_MARKET order for {symbol}: order_id={sl_order['id']}, "
                            f"stopPrice={self.sl_price[symbol]:.4f}, side={'sell' if side == 'long' else 'buy'}, qty={qty}"
                        )
                        break
                    except Exception as e:
                        logger.error(f"Failed to place STOP_MARKET for {symbol} (attempt {sl_attempt}): {e}")
                        if sl_attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place STOP_MARKET for {symbol} after {max_attempts} attempts")
                            await self.close_position(symbol, price)
                            return

                for trailing_attempt in range(1, max_attempts + 1):
                    try:
                        activation_price = price * (1 + self.trailing_activation_pct / 100 if side == 'long' else 1 - self.trailing_activation_pct / 100)
                        activation_price = round(activation_price, self.price_precision[symbol])
                        trailing_order = await self.client.exchange.create_order(
                            symbol=symbol,
                            type='TRAILING_STOP_MARKET',
                            side='sell' if side == 'long' else 'buy',
                            amount=qty,
                            params={
                                "activationPrice": activation_price,
                                "callbackRate": self.trailing_callback_rate,
                                "reduceOnly": True
                            }
                        )
                        self.trailing_order_id[symbol] = trailing_order['id']
                        logger.info(
                            f"Placed TRAILING_STOP_MARKET order for {symbol}: order_id={trailing_order['id']}, "
                            f"activationPrice={activation_price:.4f}, callbackRate={self.trailing_callback_rate}, qty={qty}"
                        )
                        break
                    except Exception as e:
                        logger.error(f"Failed to place TRAILING_STOP_MARKET for {symbol} (attempt {trailing_attempt}): {e}")
                        if trailing_attempt < max_attempts:
                            await asyncio.sleep(1)
                        else:
                            logger.error(f"Failed to place TRAILING_STOP_MARKET for {symbol} after {max_attempts} attempts")
                            await self.close_position(symbol, price)
                            return

                return

            except Exception as e:
                logger.error(f"Failed to place order for {symbol} (attempt {attempt}): {e}")
                attempt += 1
                if attempt <= max_attempts:
                    await asyncio.sleep(1)
                continue

        logger.warning(f"Failed to open {side} position for {symbol} after {max_attempts} attempts")

    async def close_position(self, symbol, price):
        try:
            side = self.position_side[symbol]
            qty = self.quantity[symbol]
            if qty <= 0:
                logger.warning(f"No quantity to close for {symbol}")
                return
            if price <= 0:
                try:
                    ticker = await self.client.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    logger.info(f"Fetched current price for {symbol} close: {price}")
                except Exception as e:
                    logger.error(f"Failed to fetch close price for {symbol}: {e}")
                    return

            for order_id in [self.tp_order_id[symbol], self.sl_order_id[symbol], self.trailing_order_id[symbol]]:
                if order_id:
                    try:
                        await self.client.exchange.cancel_order(order_id, symbol)
                        logger.info(f"Canceled order for {symbol}: order_id={order_id}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order_id} for {symbol}: {e}")

            order_side = 'sell' if side == "long" else 'buy'
            if self.config.get('mode') == 'taker':
                # Market close mode: immediately close position at market price
                try:
                    await self.client.exchange.create_market_order(symbol, order_side, qty)
                    logger.info(f"Closed {side} position for {symbol} with MARKET order, qty={qty}")
                except Exception as e:
                    logger.error(f"Failed to close position for {symbol} at market: {e}")
                    return
                # Update position state after market close
                self.in_position[symbol] = False
                self.position_side[symbol] = None
                self.entry_price[symbol] = 0.0
                self.sl_price[symbol] = 0.0
                self.tp_price[symbol] = 0.0
                self.quantity[symbol] = 0.0
                self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + self.unrealized_pnl.get(symbol, 0.0)
                self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, self.entry_price[symbol], price, self.unrealized_pnl.get(symbol, 0.0), "CLOSE")
                self.position_timeframe[symbol] = None
                self.tp_order_id[symbol] = None
                self.sl_order_id[symbol] = None
                self.trailing_order_id[symbol] = None
                return
            max_attempts = 2
            attempt = 1

            while attempt <= max_attempts:
                try:
                    offset_pct = self.price_offset_pct
                    if attempt > 1:
                        offset_pct = self.retry_offset_pct or offset_pct * 0.5
                        logger.info(f"Retry close attempt {attempt} for {symbol}: adjusted offset_pct to {offset_pct}")

                    limit_price = price * (1 + offset_pct / 100) if side == "long" else price * (1 - offset_pct / 100)
                    limit_price = round(limit_price, self.price_precision[symbol])
                    price_diff = abs(price - limit_price)
                    logger.info(
                        f"Placing MAKER CLOSE {side.upper()} LIMIT order for {symbol} (attempt {attempt}): "
                        f"qty={qty}, limit_price={limit_price:.4f}, offset_pct={offset_pct}, price_diff={price_diff:.4f}"
                    )

                    order = await self.client.exchange.create_limit_order(symbol, order_side, qty, limit_price, params={"postOnly": True})

                    wait_time = self.entry_timeout_sec
                    interval = 0.2
                    checks = int(wait_time / interval)

                    for _ in range(checks):
                        await asyncio.sleep(interval)
                        order_status = await self.client.exchange.fetch_order(order['id'], symbol)
                        if order_status['status'] == 'closed':
                            break
                    else:
                        await self.client.exchange.cancel_order(order['id'], symbol)
                        logger.info(
                            f"Canceled unfilled close order for {symbol} (attempt {attempt}): "
                            f"order_id={order['id']}, limit_price={limit_price:.4f}"
                        )
                        attempt += 1
                        price = self.latest_close[symbol]
                        if price <= 0:
                            try:
                                ticker = await self.client.exchange.fetch_ticker(symbol)
                                price = ticker['last']
                                logger.info(f"Fetched current price for {symbol} close retry: {price}")
                            except Exception as e:
                                logger.error(f"Failed to fetch close retry price for {symbol}: {e}")
                                break
                        continue

                    logger.info(f"Closed {side} {symbol} {qty} @ {price}")
                    self.in_position[symbol] = False
                    self.position_side[symbol] = None
                    self.entry_price[symbol] = 0.0
                    self.sl_price[symbol] = 0.0
                    self.tp_price[symbol] = 0.0
                    self.quantity[symbol] = 0.0
                    self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0.0) + self.unrealized_pnl.get(symbol, 0.0)
                    self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, self.entry_price[symbol], price, self.unrealized_pnl.get(symbol, 0.0), "CLOSE")
                    self.position_timeframe[symbol] = None
                    self.tp_order_id[symbol] = None
                    self.sl_order_id[symbol] = None
                    self.trailing_order_id[symbol] = None
                    return

                except Exception as e:
                    logger.error(f"Failed to close order for {symbol} (attempt {attempt}): {e}")
                    attempt += 1
                    if attempt <= max_attempts:
                        await asyncio.sleep(1)
                    continue

            logger.warning(f"Failed to close {side} position for {symbol} after {max_attempts} attempts")

        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")

    def log_trade(self, symbol, timeframe, side, entry_price, exit_price, pnl, status):
        if self.signal_priority:
            return
        result = "win" if isinstance(pnl, (float, int)) and pnl > 0 else "loss" if status == "CLOSE" else "-"
        try:
            with open("trade_log.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{symbol},{timeframe},{side},{entry_price},{exit_price or '-'},{pnl},{result},{status}\n")
        except Exception as e:
            logger.warning(f"Failed to log trade for {symbol}: {e}")

    def get_recent_trades(self, symbol):
        try:
            df = pd.read_csv("trade_log.csv", header=None, names=[
                "datetime", "symbol", "timeframe", "side", "entry", "exit", "pnl", "result", "status"
            ])
            recent = df[(df["symbol"] == symbol) & (df["status"] == "CLOSE")].tail(self.performance_window)
            return recent
        except Exception:
            return pd.DataFrame()

    def get_win_rate(self, symbol):
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return None
        return (trades['result'] == "win").mean()

    def automate_learning(self, symbol):
        if self.signal_priority:
            return
        now = datetime.datetime.utcnow()
        if (now - self.last_adaptation).total_seconds() < 600:
            return
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return
        grouped = trades.groupby("timeframe")["result"].value_counts().unstack(fill_value=0)
        for tf in self.timeframes:
            wins = grouped.loc[tf]["win"] if tf in grouped.index and "win" in grouped.columns else 0
            total = grouped.loc[tf].sum() if tf in grouped.index else 0
            win_rate = wins / total if total > 0 else 0
            if total >= 8 and win_rate < self.min_win_rate:
                self.disable_timeframes[symbol].add(tf)
                logger.info(f"Disabled {tf} for {symbol} (win rate {win_rate:.2f})")
            elif tf in self.disable_timeframes[symbol] and win_rate >= self.min_win_rate:
                self.disable_timeframes[symbol].remove(tf)
                logger.info(f"Re-enabled {tf} for {symbol} (win rate improved to {win_rate:.2f})")
        self.last_adaptation = now

    def display_status(self):
        if self.signal_priority:
            return
        table = Table(title=f"Strategy Status ({self.leverage}x Leverage)", show_lines=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Side", style="magenta")
        table.add_column("Entry Price", style="yellow")
        table.add_column("Last Price", style="white")
        table.add_column("Stop Loss", style="red")
        table.add_column("Take Profit", style="blue")
        table.add_column("Qty", style="cyan")
        table.add_column("Unreal. PnL", style="green")
        table.add_column("PnL %", style="green")
        table.add_column("Realiz. PnL", style="green")
        table.add_column("Liq. Price", style="red")
        table.add_column("Margin", style="yellow")
        table.add_column("Mgn %", style="yellow")
        table.add_column("Funding", style="yellow")
        table.add_column("Comm.", style="red")
        table.add_column("Tfs OFF", style="bold red")
        table.add_column("TP PnL", style="green")
        table.add_column("SL PnL", style="red")
        for symbol in self.symbols:
            pos = "IN POSITION" if self.in_position.get(symbol) else "NO POSITION"
            side = self.position_side.get(symbol) or "-"
            entry = self.entry_price.get(symbol, 0.0)
            price = self.latest_close.get(symbol, 0.0)
            sl = self.sl_price.get(symbol, 0.0)
            tp = self.tp_price.get(symbol, 0.0)
            qty = self.quantity.get(symbol, 0.0)
            unpnl = self.unrealized_pnl.get(symbol, 0.0)
            margin = self.margin_used.get(symbol, 0.0)
            pnl_pct = f"{(unpnl/margin*100):.2f}%" if margin else "0.00%"
            rpnl = f"{self.realized_pnl.get(symbol, 0.0):.4f}"
            liq = f"{self.liquidation_price.get(symbol, 0.0):.2f}"
            mgn = f"{margin:.2f}"
            mgn_pct = f"{self.margin_ratio.get(symbol, 0.0):.2f}%"
            fund = f"{self.funding_fee.get(symbol, 0.0):.4f}"
            comm = f"{self.commission.get(symbol, 0.0):.4f}"
            tfs_off = ",".join(self.disable_timeframes[symbol]) if self.disable_timeframes[symbol] else "-"

            if qty > 0:
                if side == "long":
                    pnl_tp = (tp - entry) * qty
                    pnl_sl = (sl - entry) * qty
                elif side == "short":
                    pnl_tp = (entry - tp) * qty
                    pnl_sl = (entry - sl) * qty
                else:
                    pnl_tp = pnl_sl = 0.0
            else:
                pnl_tp = pnl_sl = 0.0

            table.add_row(
                symbol, pos, side, f"{entry:.2f}", f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}", f"{qty:.4f}", f"{unpnl:.4f}",
                pnl_pct, rpnl, liq, mgn, mgn_pct, fund, comm, tfs_off,
                f"{pnl_tp:.2f}", f"{pnl_sl:.2f}"
            )
        self.console.print(table)

    def enable_force_entry(self):
        self.force_entry = True
        logger.warning("Force entry is ENABLED: bot will always try to enter a long position for testing.")

    def disable_force_entry(self):
        self.force_entry = False
        logger.warning("Force entry is DISABLED: bot will only enter on real signals.")

    def print_signals(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                signal = self.get_signal_for_timeframe(symbol, tf)
                print(f"{symbol} {tf}: {signal}")
