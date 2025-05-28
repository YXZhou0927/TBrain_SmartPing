import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 features 模組")

from . import timeseries_features
#from .feature_selector import select_top_features

__all__ = ["timeseries_features"]
__version__ = "0.1.0"
