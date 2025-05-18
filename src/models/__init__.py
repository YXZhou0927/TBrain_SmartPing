import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("📦 載入 models 模組")

#from .train_model import train_model
#from .predict_model import predict
#from .evaluate_model import evaluate

# Quickly train, evaluate and predict
import models.one_click

__all__ = ["one_click"]
__version__ = "0.1.0"
