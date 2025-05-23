import logging

logger = logging.getLogger(__name__)

class IndicatorOptimizer:
    def __init__(self, strategy, backtest_days=30):
        self.strategy = strategy
        self.backtest_days = backtest_days
        self.last_optimization = None

    def should_optimize(self):
        return False

    def optimize(self):
        logger.info("IndicatorOptimizer: stub, no optimization performed.")
