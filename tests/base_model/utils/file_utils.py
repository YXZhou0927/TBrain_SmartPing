"""
檔案操作相關工具函數
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
import xgboost as xgb  # 導入xgboost用於保存模型

def save_study(study, model_name, model_dir):
    """保存Optuna study物件
    
    Args:
        study: Optuna study物件
        model_name (str): 模型名稱
        model_dir (str): 模型目錄
    """
    study_path = os.path.join(model_dir, f"{model_name}_study.pkl")
    joblib.dump(study, study_path)
    print(f"Study已保存至: {study_path}")

def save_params(params, model_name, model_dir):
    """保存模型參數
    
    Args:
        params (dict): 參數字典
        model_name (str): 模型名稱
        model_dir (str): 模型目錄
    """
    params_path = os.path.join(model_dir, f"{model_name}_best_params.pkl")
    joblib.dump(params, params_path)
    print(f"最佳參數已保存至: {params_path}")

def save_scaler(scaler, model_name, scaler_type, model_dir):
    """保存標準化器
    
    Args:
        scaler: 標準化器物件
        model_name (str): 模型名稱
        scaler_type (str): 標準化方法
        model_dir (str): 模型目錄
    """
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"{model_name}_best_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"標準化器({scaler_type})已保存至: {scaler_path}")

def save_model(model, model_name, fold_idx, model_dir):
    """保存單個fold的模型
    
    Args:
        model: xgb.Booster 模型物件
        model_name (str): 模型名稱
        fold_idx (int): fold索引
        model_dir (str): 模型目錄
    """
    model_path = os.path.join(model_dir, f"{model_name}_fold{fold_idx}_model.json")
    model.save_model(model_path)
    print(f"第{fold_idx}折模型已保存至: {model_path}")

def save_model_info(model_info, model_name, model_dir):
    """保存模型資訊
    
    Args:
        model_info (dict): 模型資訊字典
        model_name (str): 模型名稱
        model_dir (str): 模型目錄
    """
    model_info_path = os.path.join(model_dir, f"{model_name}_model_info.pkl")
    joblib.dump(model_info, model_info_path)
    print(f"模型整合資訊已保存至: {model_info_path}")

def save_meta_features(meta_features, model_type, meta_dir, is_test=False):
    """保存meta-features
    
    Args:
        meta_features (pd.DataFrame): meta-features DataFrame
        model_type (str): 模型類型
        meta_dir (str): meta-features目錄
        is_test (bool): 是否為測試資料
    """
    prefix = "test" if is_test else "train"
    meta_file = os.path.join(meta_dir, f"{model_type}_{prefix}_meta_features.csv")
    meta_features.to_csv(meta_file, index=False)
    print(f"{prefix}資料的Meta-features保存到: {meta_file}")

def save_submission(submission, model_type, submission_dir):
    """保存提交檔案
    
    Args:
        submission (pd.DataFrame): 提交DataFrame
        model_type (str): 模型類型
        submission_dir (str): 提交目錄
    """
    submission_path = os.path.join(submission_dir, f"{model_type}_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"提交檔案保存到: {submission_path}")

def load_model_info(model_type, task, model_dir):
    """載入模型資訊
    
    Args:
        model_type (str): 模型類型
        task (str): 任務名稱
        model_dir (str): 模型目錄
        
    Returns:
        dict: 模型資訊字典
    """
    model_info_path = os.path.join(model_dir, f"{model_type}_{task}_model_info.pkl")
    if os.path.exists(model_info_path):
        return joblib.load(model_info_path)
    else:
        print(f"警告: 找不到模型資訊 {model_info_path}")
        return None

def load_scaler(model_type, task, model_dir):
    """載入標準化器
    
    Args:
        model_type (str): 模型類型
        task (str): 任務名稱
        model_dir (str): 模型目錄
        
    Returns:
        object: 標準化器
    """
    scaler_path = os.path.join(model_dir, f"{model_type}_{task}_best_scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        print(f"警告: 找不到標準化器 {scaler_path}")
        return None

def load_study(model_type, task, model_dir):
    """載入Optuna study物件
    
    Args:
        model_type (str): 模型類型
        task (str): 任務名稱
        model_dir (str): 模型目錄
        
    Returns:
        object: Study物件
    """
    study_path = os.path.join(model_dir, f"{model_type}_{task}_study.pkl")
    if os.path.exists(study_path):
        return joblib.load(study_path)
    else:
        print(f"警告: 找不到study物件 {study_path}")
        return None

def load_models(model_paths):
    """載入多個模型
    
    Args:
        model_paths (list): 模型路徑列表
        
    Returns:
        list: 模型列表
    """
    models = []
    for path in model_paths:
        if os.path.exists(path):
            model = xgb.Booster()
            model.load_model(path)
            models.append(model)
        else:
            print(f"警告: 找不到模型檔案 {path}")
    return models

def merge_submissions(submission_files, output_path, sample_submission_path):
    """合併多個任務的提交檔案
    
    Args:
        submission_files (dict): 任務名稱到檔案路徑的字典
        output_path (str): 輸出檔案路徑
        sample_submission_path (str): 樣本提交檔案路徑
        
    Returns:
        pd.DataFrame: 合併後的提交檔案
    """
    # 載入樣本提交檔案以獲取正確的格式和順序
    sample_submission = pd.read_csv(sample_submission_path)
    
    # 初始化合併後的提交檔案
    merged_submission = pd.DataFrame({'unique_id': sample_submission['unique_id']})
    
    # 逐個載入和合併提交檔案
    for task, file_path in submission_files.items():
        if os.path.exists(file_path):
            task_submission = pd.read_csv(file_path)
            
            # 獲取需要合併的列（排除unique_id）
            merge_columns = [col for col in task_submission.columns if col != 'unique_id']
            
            # 確保順序正確
            task_submission = task_submission.set_index('unique_id')
            
            # 合併到結果中
            for col in merge_columns:
                if col in sample_submission.columns:
                    merged_submission[col] = task_submission.loc[merged_submission['unique_id']][col].values
        else:
            print(f"警告: 找不到提交檔案 {file_path}")
    
    # 保存合併後的提交檔案
    merged_submission.to_csv(output_path, index=False)
    print(f"合併提交檔案已保存至: {output_path}")
    
    return merged_submission