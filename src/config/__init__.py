import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 config 模組")

from .data_config_manager import DataConfigManager
from .xgbtrainer_config import XGBTrainerConfig

__all__ = ["DataConfigManager", "XGBTrainerConfig"]
__version__ = "0.1.0"