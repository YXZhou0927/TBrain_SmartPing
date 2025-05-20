"""
基礎配置文件，包含所有任務共用的配置參數
"""

import os

# 創建結果目錄
BASE_RESULTS_DIR = "results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# 資料路徑
TRAIN_INFO_PATH = 'train_info.csv'
TEST_INFO_PATH = 'test_info.csv'
TRAIN_DATA_DIR = './tabular_data_train'
TEST_DATA_DIR = './tabular_data_test'
SAMPLE_SUBMISSION_PATH = 'sample_submission.csv'

# 模型設定
USE_GPU = True
DEFAULT_MODEL_TYPE = "xgb"

# 任務列表
TASKS = ['gender', 'hold_racket_handed', 'play_years', 'level']

# 獲取任務結果目錄
def get_task_dir(task_name):
    """獲取特定任務的結果目錄
    
    Args:
        task_name (str): 任務名稱
        
    Returns:
        tuple: (任務目錄, 模型目錄, meta特徵目錄, 提交目錄)
    """
    task_dir = os.path.join(BASE_RESULTS_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # 創建子目錄
    models_dir = os.path.join(task_dir, "models")
    meta_dir = os.path.join(task_dir, "meta_features")
    submission_dir = os.path.join(task_dir, "submissions")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    return task_dir, models_dir, meta_dir, submission_dir