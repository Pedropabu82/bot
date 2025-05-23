
import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import joblib

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
    return features.dropna()

def train_model(log_path='trade_log.csv', model_output='model_xgb.pkl'):
    try:
        trades = pd.read_csv(log_path)
        trades = trades.dropna()
        trades = trades[trades['type'] == 'ENTRY']

        dfs = []
        labels = []
        for _, row in trades.iterrows():
            try:
                ohlcv = pd.read_csv(f"data/{row['symbol']}_{row['timeframe']}.csv")
                ohlcv.columns = ['timestamp','open','high','low','close','volume']
                ohlcv = ohlcv.tail(150).reset_index(drop=True)
                feats = extract_features(ohlcv)
                if feats.empty:
                    continue
                dfs.append(feats.iloc[-1])
                labels.append(1 if row['result'].lower() == 'win' else 0)
            except Exception as e:
                print(f"Erro ao processar {row['symbol']} {row['timeframe']}: {e}")

        if not dfs:
            print("Nenhum dado válido para treinar.")
            return

        X = pd.DataFrame(dfs)
        y = np.array(labels)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, model_output)
        print(f"Modelo treinado e salvo em: {model_output}")
    except Exception as e:
        print(f"Erro ao treinar modelo: {e}")

if __name__ == '__main__':
    train_model()
