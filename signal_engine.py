
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class SignalEngine:
    def __init__(self, model_path="model_xgb.pkl"):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("SignalEngine: Modelo XGBoost carregado com sucesso.")
            else:
                logger.warning("SignalEngine: Modelo não encontrado, fallback ativado.")
        except Exception as e:
            logger.warning(f"SignalEngine: Falha ao carregar modelo ({e}); fallback ativado.")

    def get_signal_for_timeframe(self, data, min_score=0.65, symbol=None, timeframe=None):
        try:
            # Calcular score simples baseado nos indicadores (fallback)
            base_score = sum([
                data.get('ema', 0),
                data.get('macd', 0),
                data.get('rsi', 0),
                data.get('adx', 0),
                data.get('obv', 0),
                data.get('atr', 0),
                data.get('volume', 0)
            ]) / 7

            proba = base_score
            ok = base_score >= min_score
            decision = "✅" if ok else "❌"

            # Se modelo carregado, usa predição real
            if self.model:
                features = [[
                    data.get('ema', 0),
                    data.get('macd', 0),
                    data.get('rsi', 0),
                    data.get('adx', 0),
                    data.get('obv', 0),
                    data.get('atr', 0),
                    data.get('volume', 0)
                ]]
                proba = self.model.predict_proba(features)[0][1]
                ok = proba >= 0.5
                decision = "✅" if ok else "❌"

            logger.info(f"[AI] {symbol or ''} {timeframe or ''} - proba: {proba:.4f} - decision: {decision}")
            return {"ok": ok, "confidence": round(proba, 4)}

        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return {"ok": False, "confidence": 0.0}
