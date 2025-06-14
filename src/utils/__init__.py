import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 utils 工具模組")

from .rename_df import rename_df_columns
# from .logger import setup_logger
# from .path import get_project_root
# from .timer import Timer

__all__ = ["rename_df_columns"]
__version__ = "0.1.0"
