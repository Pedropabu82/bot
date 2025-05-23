
import pandas as pd
import numpy as np
import ccxt
import joblib
import xgboost as xgb
import talib
import time
import os

def fetch_ohlcv(symbol, timeframe, since, limit=300):
    binance = ccxt.binance({
        'enableRateLimit': True,
    })
    binance.options['defaultType'] = 'future'
    try:
        data = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Erro ao buscar candles para {symbol} {timeframe}: {e}")
        return None

def extract_features(df):
    features = pd.DataFrame()
    features['ema_short'] = talib.EMA(df['close'], timeperiod=9)
    features['ema_long'] = talib.EMA(df['close'], timeperiod=21)
    macd, macdsignal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd'] = macd
    features['macdsignal'] = macdsignal
    features['rsi'] = talib.RSI(df['close'], timeperiod=14)
    features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    features['obv'] = talib.OBV(df['close'], df['volume'])
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['volume'] = df['volume']
    return features.dropna().reset_index(drop=True)

def train_from_log(trade_log='trade_log.csv'):
    if not os.path.exists(trade_log):
        print(f"Arquivo {trade_log} não encontrado.")
        return

    trades = pd.read_csv(trade_log)
    trades = trades.dropna()
    trades = trades[trades['type'] == 'ENTRY']
    print(f"Carregando {len(trades)} trades do log.")

    X, y = [], []

    for _, row in trades.iterrows():
        symbol = row['symbol']
        timeframe = row['timeframe']
        timestamp = pd.to_datetime(row['timestamp'])
        since = int((timestamp - pd.Timedelta(minutes=600)).timestamp() * 1000)

        df = fetch_ohlcv(symbol, timeframe, since)
        if df is None or df.empty:
            print(f"Dados vazios para {symbol} {timeframe}, pulando...")
            continue

        feats = extract_features(df)
        if feats.empty:
            print(f"Features vazias para {symbol} {timeframe}, pulando...")
            continue

        X.append(feats.iloc[-1])
        y.append(1 if row['result'].lower() == 'win' else 0)

        time.sleep(0.1)

    if not X:
        print("Nenhum dado válido coletado para treino.")
        return

    unique_classes = set(y)
    print(f"Classes encontradas no y: {unique_classes}")
    if len(unique_classes) < 2:
        print("❌ ERRO: Apenas uma classe detectada no vetor y. Adicione mais trades de tipos diferentes (win/loss).")
        return

    df_X = pd.DataFrame(X)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(df_X, y)
    joblib.dump(model, "model_xgb.pkl")
    print("✅ Modelo treinado e salvo como model_xgb.pkl")

if __name__ == "__main__":
    train_from_log()
