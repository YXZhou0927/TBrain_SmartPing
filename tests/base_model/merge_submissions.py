"""
合併所有任務的提交文件
"""

import os
import pandas as pd
from config import base_config
from config import gender_config, hold_racket_config, play_years_config, level_config
from utils import file_utils

def merge_all_submissions():
    """
    合併所有任務的提交文件為一個完整的提交文件
    
    Returns:
        pd.DataFrame: 合併後的提交文件
    """
    print("\n開始合併所有任務的提交文件...")
    
    # 獲取樣本提交以確保格式和順序正確
    sample_submission = pd.read_csv(base_config.SAMPLE_SUBMISSION_PATH)
    
    # 各任務提交文件路徑
    submission_files = {
        'gender': os.path.join(gender_config.SUBMISSION_DIR, "xgb_submission.csv"),
        'hold_racket': os.path.join(hold_racket_config.SUBMISSION_DIR, "xgb_submission.csv"),
        'play_years': os.path.join(play_years_config.SUBMISSION_DIR, "xgb_submission.csv"),
        'level': os.path.join(level_config.SUBMISSION_DIR, "xgb_submission.csv")
    }
    
    # 檢查文件是否存在
    for task, file_path in submission_files.items():
        if not os.path.exists(file_path):
            print(f"警告: {task} 任務的提交文件 {file_path} 不存在")
    
    # 初始化合併後的提交文件
    merged_submission = pd.DataFrame({'unique_id': sample_submission['unique_id']})
    
    # 逐個載入和合併提交文件
    try:
        # 合併性別預測
        if os.path.exists(submission_files['gender']):
            gender_submission = pd.read_csv(submission_files['gender'])
            merged_submission['gender'] = gender_submission.set_index('unique_id').loc[merged_submission['unique_id']]['gender'].values
        
        # 合併慣用手預測
        if os.path.exists(submission_files['hold_racket']):
            hand_submission = pd.read_csv(submission_files['hold_racket'])
            merged_submission['hold racket handed'] = hand_submission.set_index('unique_id').loc[merged_submission['unique_id']]['hold racket handed'].values
        
        # 合併打球年限預測
        if os.path.exists(submission_files['play_years']):
            years_submission = pd.read_csv(submission_files['play_years'])
            for col in [col for col in sample_submission.columns if col.startswith('play years_')]:
                merged_submission[col] = years_submission.set_index('unique_id').loc[merged_submission['unique_id']][col].values
        
        # 合併水平級別預測
        if os.path.exists(submission_files['level']):
            level_submission = pd.read_csv(submission_files['level'])
            for col in [col for col in sample_submission.columns if col.startswith('level_')]:
                merged_submission[col] = level_submission.set_index('unique_id').loc[merged_submission['unique_id']][col].values
        
        # 保存合併後的提交文件
        output_path = os.path.join(base_config.BASE_RESULTS_DIR, "final_submission.csv")
        merged_submission.to_csv(output_path, index=False)
        print(f"合併完成，最終提交文件保存至: {output_path}")
        
        # 驗證合併後的文件
        verify_submission(merged_submission, sample_submission)
        
        return merged_submission
        
    except Exception as e:
        print(f"合併提交文件時出錯: {str(e)}")
        return None

def verify_submission(submission, sample_submission):
    """
    驗證合併後的提交文件是否符合要求
    
    Args:
        submission (pd.DataFrame): 合併後的提交文件
        sample_submission (pd.DataFrame): 樣本提交文件
    """
    # 檢查列是否一致
    missing_cols = set(sample_submission.columns) - set(submission.columns)
    extra_cols = set(submission.columns) - set(sample_submission.columns)
    
    if missing_cols:
        print(f"警告: 合併提交文件缺少以下列: {missing_cols}")
    
    if extra_cols:
        print(f"警告: 合併提交文件包含多餘的列: {extra_cols}")
    
    # 檢查行數是否一致
    if len(submission) != len(sample_submission):
        print(f"警告: 合併提交文件的行數 ({len(submission)}) 與樣本提交文件的行數 ({len(sample_submission)}) 不一致")
    
    # 檢查概率值是否有效
    for col in submission.columns:
        if col == 'unique_id':
            continue
        
        values = submission[col].values
        if (values < 0).any() or (values > 1).any():
            print(f"警告: 列 {col} 包含無效的概率值 (不在 [0, 1] 範圍內)")
    
    print("提交文件驗證完成")

if __name__ == "__main__":
    merge_all_submissions()