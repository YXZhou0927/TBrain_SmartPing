"""
資料處理相關工具函數
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer

def get_scaler(scaler_type):
    """根據給定類型返回標準化器實例
    
    Args:
        scaler_type (str): 標準化方法名稱
        
    Returns:
        object: 標準化器實例
    """
    if scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "maxabs":
        return MaxAbsScaler()
    elif scaler_type == "yeo-johnson":
        return PowerTransformer(method='yeo-johnson')
    else:
        # 默認使用MinMaxScaler
        return MinMaxScaler()

def prepare_data(info, datapath, is_test=False):
    """統一的資料準備函數，處理特徵、標籤和ID
    
    Args:
        info (pd.DataFrame): 資料信息DataFrame
        datapath (str): 資料路徑
        is_test (bool): 是否為測試資料
        
    Returns:
        tuple: 組合後的特徵、標籤和unique_id，以及mode特徵索引
    """
    datalist = list(Path(datapath).glob('**/*.csv'))
    X_data = []
    y_data = []
    unique_ids = []
    player_ids = []
    mode_values = []  # 收集mode值
    
    mode_feature_idx = 0  # mode是第一個特徵
    
    for file in datalist:
        unique_id = int(Path(file).stem)
        
        # 如果是訓練資料，需要檢查是否在info中
        if not info.empty:
            row = info[info['unique_id'] == unique_id]
            if row.empty:
                continue
            
            # 收集mode值
            mode_value = row['mode'].iloc[0]
            mode_values.append(mode_value)
        
        data = pd.read_csv(file)
        features = data.values.flatten()
        
        if not info.empty and not is_test:  # 訓練資料
            labels = {
                'gender': row['gender'].iloc[0],
                'hold racket handed': row['hold racket handed'].iloc[0],
                'play years': row['play years'].iloc[0],
                'level': row['level'].iloc[0]
            }
            y_data.append(labels)
            player_ids.append(row['player_id'].iloc[0])
        
        X_data.append(features)
        unique_ids.append(unique_id)
    
    # 將原始特徵轉換為numpy陣列
    X_original = np.array(X_data)
    
    # 處理mode特徵 - 保持為類別特徵
    if not info.empty:
        mode_values = np.array(mode_values).reshape(-1, 1)
        
        # 將mode特徵與原始特徵結合
        X_combined = np.column_stack((mode_values, X_original))
        print(f"結合後的特徵維度: {X_combined.shape}")
    else:
        # 如果沒有info資訊，就只返回原始特徵
        X_combined = X_original
        print(f"沒有mode資訊，使用原始特徵，維度: {X_combined.shape}")
    
    if not info.empty and not is_test:
        y_df = pd.DataFrame(y_data, index=unique_ids)
        y_df['player_id'] = player_ids
        return X_combined, y_df, unique_ids, mode_feature_idx
    else:
        return X_combined, unique_ids, mode_feature_idx

def transform_labels(y, task):
    """將標籤轉換為從0開始的連續整數
    
    Args:
        y (np.array): 原始標籤
        task (str): 任務名稱
        
    Returns:
        np.array: 轉換後的標籤
    """
    y_transformed = y.copy()
    if task == 'gender' or task == 'hold_racket_handed':
        # 將標籤1,2轉換為0,1
        y_transformed = y_transformed - 1
    elif task == 'level':
        # 將標籤2,3,4,5轉換為0,1,2,3
        y_transformed = y_transformed - 2
    # play_years已經是從0開始的無需轉換
    return y_transformed

def split_features_by_mode(X, mode_feature_idx):
    """將特徵按mode特徵拆分
    
    Args:
        X (np.array): 組合後的特徵
        mode_feature_idx (int): mode特徵的索引
        
    Returns:
        tuple: mode特徵和其他特徵
    """
    mode_feature = X[:, mode_feature_idx:mode_feature_idx+1].copy()
    other_features = X[:, mode_feature_idx+1:].copy()
    return mode_feature, other_features

def apply_scaling(mode_feature, other_features, scaler):
    """應用標準化並重新組合特徵
    
    Args:
        mode_feature (np.array): mode特徵
        other_features (np.array): 其他特徵
        scaler: 標準化器
        
    Returns:
        np.array: 標準化後的特徵
    """
    if scaler is not None:
        scaled_features = scaler.transform(other_features)
        # 將mode特徵與標準化後的特徵重新組合
        X_scaled = np.column_stack((mode_feature, scaled_features))
    else:
        X_scaled = np.column_stack((mode_feature, other_features))
    return X_scaled

def create_feature_types(X_shape, categorical_features):
    """創建特徵類型列表
    
    Args:
        X_shape (tuple): 特徵矩陣形狀
        categorical_features (list): 類別特徵索引列表
        
    Returns:
        list: 特徵類型列表
    """
    return ['c' if i in categorical_features else 'q' for i in range(X_shape[1])]

def combine_predictions(predictions_list, weights=None):
    """組合多個模型的預測結果
    
    Args:
        predictions_list (list): 預測結果列表
        weights (list, optional): 權重列表，若不指定則使用平均權重
        
    Returns:
        np.array: 組合後的預測結果
    """
    if not predictions_list:
        return None
    
    n_samples = predictions_list[0].shape[0]
    n_classes = predictions_list[0].shape[1]
    
    if weights is None:
        # 使用相等權重
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    else:
        # 確保權重和為1
        weights = np.array(weights) / np.sum(weights)
    
    # 初始化組合預測
    combined_pred = np.zeros((n_samples, n_classes))
    
    # 加權組合預測
    for i, pred in enumerate(predictions_list):
        combined_pred += weights[i] * pred
    
    return combined_pred