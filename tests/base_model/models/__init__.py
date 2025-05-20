"""
模型模組初始化
"""

# 導入所有模型類以便直接從models包中引用
from models.base_model import BaseMetaFeaturesGenerator, XGBoostBase
from models.gender_model import GenderXGBGenerator
from models.hold_racket_model import HoldRacketXGBGenerator
from models.play_years_model import PlayYearsXGBGenerator
from models.level_model import LevelXGBGenerator

# 創建模型工廠方法，可通過名稱創建模型實例
def create_model(task_name, use_gpu=True):
    """
    根據任務名稱創建相應的模型實例
    
    Args:
        task_name (str): 任務名稱
        use_gpu (bool): 是否使用GPU
        
    Returns:
        BaseMetaFeaturesGenerator: 模型實例
    """
    if task_name == 'gender':
        return GenderXGBGenerator(use_gpu=use_gpu)
    elif task_name == 'hold_racket_handed':
        return HoldRacketXGBGenerator(use_gpu=use_gpu)
    elif task_name == 'play_years':
        return PlayYearsXGBGenerator(use_gpu=use_gpu)
    elif task_name == 'level':
        return LevelXGBGenerator(use_gpu=use_gpu)
    else:
        raise ValueError(f"未知的任務名稱: {task_name}")