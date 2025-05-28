import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 data 模組")

from .load_data import load_data
from .load_data import merge_metadata_and_features
#from .save_data import save_data
#from .check_data import check_data

__all__ = ["load_data", "merge_metadata_and_features"]
__version__ = "0.1.0"
