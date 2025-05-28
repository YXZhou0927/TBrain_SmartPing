import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
from ..config import DataConfigManager
from typing import Union, List
from pathlib import Path


def process_info_df(info_df: pd.DataFrame, target_col=None):
    """
    處理輸入的 DataFrame，提取 ID、特徵和目標值。
    這個函數會檢查 DataFrame 是否包含 'unique_id' 和 'mode' 列，並且提取以 'feature_' 開頭的列作為特徵。
    如果提供了目標列名稱，則提取該列作為目標值。
    這個函數返回三個 DataFrame：一個包含 ID 的 DataFrame，一個包含特徵的 DataFrame，和一個包含目標值的 DataFrame（如果提供了目標列名稱）。
    這個函數的主要目的是為了方便後續模型訓練和預測的數據處理。
    Args:
        info_df (pd.DataFrame): 包含 unique_id 的 DataFrame
        target_col (str, optional): 目標列的名稱，默認為None
    Returns:
        uid_df (pd.DataFrame): 包含ID相關資訊的DataFrame
        features_df (pd.DataFrame): 包含 "feature_" 開頭的DataFrame
        target_df (pd.DataFrame): 包含目標值的DataFrame，如果未提供目標列則為None
    """
    if not isinstance(info_df, pd.DataFrame):
        raise ValueError("info_df must be a pandas DataFrame.")
    if info_df.empty:
        raise ValueError("info_df cannot be an empty DataFrame.")
    if 'unique_id' not in info_df.columns:
        raise ValueError("info_df must contain a 'unique_id' column.")
    if 'mode' not in info_df.columns:
        raise ValueError("info_df must contain a 'mode' column.")
    if target_col not in info_df.columns and target_col is not None:
        raise ValueError(f"target_col '{target_col}' not found in info_df columns.")
    # feautres columns name rule: 以"feature_"開頭的列
    features_cols = [col for col in info_df.columns if col.startswith('feature_')]
    if not features_cols:
        raise ValueError("No feature columns found in the DataFrame. Ensure columns start with 'feature_'.")
    
    if 'player_id' in info_df.columns:
        uid_df = info_df[['unique_id', 'mode', 'player_id']].copy()
    else:
        uid_df = info_df[['unique_id', 'mode']].copy()
        uid_df['player_id'] = None
    features_df = info_df[features_cols].copy()
    target_df = info_df[[target_col]].copy() if target_col is not None else None
    return uid_df, features_df, target_df

def process_predictions(preds: np.ndarray, one_hot_columns: list[str]) -> pd.DataFrame:
    """
    處理模型預測結果，將預測結果轉換為 DataFrame 格式。
    Args:
        preds (np.ndarray): 模型預測結果，通常為一維或二維數組。
        one_hot_columns (list[str]): 獨熱編碼的列名列表，用於處理多類別預測。
    Returns:
        pd.DataFrame: 包含預測結果的 DataFrame。
    """
    if not isinstance(preds, np.ndarray):
        raise ValueError("preds must be a numpy ndarray.")
    
    # 若為一維，轉為二維以便對應欄位
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if preds.shape[1] != len(one_hot_columns):
        raise ValueError(
            f"The number of columns in 'preds' ({preds.shape[1]}) does not match the length of 'one_hot_columns' ({len(one_hot_columns)})."
        )
    return pd.DataFrame(preds, columns=one_hot_columns)

def process_features(features: Union[pd.DataFrame, np.ndarray], scaler_type: str = 'minmax'):
    """
    處理特徵 DataFrame，將 mode 特徵與其他特徵分開，並應用標準化。
    
    Args:
        features (pd.DataFrame | np.ndarray): 包含特徵的 DataFrame 或 np.ndarray。
        scaler: 標準化器名稱，默認為 'minmax'。
        
    Returns:
        np.array: 處理後的特徵數組
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    elif scaler_type == "yeo-johnson":
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Supported types are: 'minmax', 'standard', 'robust', 'maxabs', 'yeo-johnson'.")

    if isinstance(features, pd.DataFrame):
        return scaler.fit_transform(features.values)
    elif isinstance(features, np.ndarray):
        return scaler.fit_transform(features)
    else:
        raise ValueError("features must be a pandas DataFrame or a numpy ndarray.")