import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 utils 工具模組")

from .logger import setup_logger
from .path import get_project_root
from .timer import Timer

__all__ = ["setup_logger", "get_project_root", "Timer"]
__version__ = "0.1.0"
