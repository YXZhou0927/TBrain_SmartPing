"""
水平級別任務執行模組
"""

import os
import time
from models import LevelXGBGenerator
from config import level_config as config
from config import base_config

def run_level_task(n_trials=None, use_gpu=None):
    """
    執行水平級別分類任務
    
    Args:
        n_trials (int, optional): Optuna試驗次數，若不指定則使用配置中的值
        use_gpu (bool, optional): 是否使用GPU，若不指定則使用配置中的值
        
    Returns:
        tuple: (提交結果, Meta-features)
    """
    task_start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"開始執行 {config.TASK_DISPLAY_NAME} 分類任務")
    print(f"{'='*50}")
    
    # 使用配置中的值或傳入的值
    if use_gpu is None:
        use_gpu = base_config.USE_GPU
    
    # 創建任務的模型生成器
    model = LevelXGBGenerator(use_gpu=use_gpu)
    
    # 優化模型並生成meta-features
    meta_features = model.optimize_and_generate_meta(n_trials=n_trials)
    
    # 預測測試數據
    submission, test_meta = model.predict_test_data()
    
    task_time = time.time() - task_start_time
    
    print(f"\n{config.TASK_DISPLAY_NAME} 任務完成！")
    print(f"耗時: {task_time:.2f} 秒")
    
    if submission is not None:
        submission_path = os.path.join(config.SUBMISSION_DIR, f"xgb_submission.csv")
        print(f"提交文件已保存至: {submission_path}")
    
    return submission, test_meta

if __name__ == "__main__":
    # 直接執行此文件時，運行水平級別任務
    run_level_task()