"""
打球年限任務特定配置
"""

import os
from config.base_config import get_task_dir

# 任務名稱
TASK_NAME = 'play_years'
TASK_COLUMN = 'play years'  # 資料表中對應的列名
TASK_DISPLAY_NAME = '打球年限'  # 顯示名稱

# 任務特定配置
CLASSES = 3  # 類別數量: 0=少於一年, 1=一到三年, 2=三年以上
CV_SPLITS = 5  # 交叉驗證折數
TRIAL_COUNT = 1  # Optuna優化的試驗次數
SCALER_TYPE = 'yeo-johnson'  # 標準化方法

# 獲取目錄
TASK_DIR, MODEL_DIR, META_FEATURES_DIR, SUBMISSION_DIR = get_task_dir(TASK_NAME)

# 模型命名
MODEL_NAME_PREFIX = f"xgb_{TASK_NAME}"