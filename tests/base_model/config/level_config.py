"""
水平級別任務特定配置
"""

import os
from config.base_config import get_task_dir

# 任務名稱
TASK_NAME = 'level'
TASK_COLUMN = 'level'  # 資料表中對應的列名
TASK_DISPLAY_NAME = '水平級別'  # 顯示名稱

# 任務特定配置
CLASSES = 4  # 類別數量: 2=初學者, 3=中級, 4=中高級, 5=高級 (需要-2轉為0,1,2,3)
CV_SPLITS = 5  # 交叉驗證折數
TRIAL_COUNT = 1  # Optuna優化的試驗次數
SCALER_TYPE = 'yeo-johnson'  # 標準化方法

# 獲取目錄
TASK_DIR, MODEL_DIR, META_FEATURES_DIR, SUBMISSION_DIR = get_task_dir(TASK_NAME)

# 模型命名
MODEL_NAME_PREFIX = f"xgb_{TASK_NAME}"