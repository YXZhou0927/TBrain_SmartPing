from sklearn.model_selection import GroupKFold, StratifiedKFold
import pandas as pd
import numpy as np

def group_kfold_by_player_id(info_df: pd.DataFrame, target_col: str, n_splits: int = 5, random_state: int = 42) -> list:
    """
    使用StratifiedKFold按player_id分割數據，返回訓練和驗證唯一ID的分割列表。
    Args:
        info_df (pd.DataFrame): 包含特徵和目標值的DataFrame
        target_col (str): 目標列名
        n_splits (int): 分割數量，默認為5
        random_state (int): 隨機種子，默認為42
    Returns:
        list: 包含訓練和驗證唯一ID的分割列表
    """
    if not isinstance(info_df, pd.DataFrame):
        raise ValueError("info_df must be a pandas DataFrame.")
    if info_df.empty:
        raise ValueError("info_df is empty. Please provide a DataFrame with data.")
    if target_col not in info_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in info_df. Please provide a valid DataFrame.")
    if 'unique_id' not in info_df.columns:
        raise ValueError("unique_id column is required in info_df for grouping.")
    if 'player_id' not in info_df.columns:
        raise ValueError("player_id column is required in info_df for grouping.")
    
    player_info = info_df.groupby("player_id")[target_col].first().reset_index()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    player_splits = list(skf.split(player_info['player_id'], player_info[target_col]))
    uid_splits = []
    for train_idx, val_idx in player_splits:
        train_players = player_info.iloc[train_idx]['player_id'].values
        val_players = player_info.iloc[val_idx]['player_id'].values
        
        train_uids = info_df[info_df['player_id'].isin(train_players)]['unique_id'].values
        val_uids = info_df[info_df['player_id'].isin(val_players)]['unique_id'].values
        
        uid_splits.append((train_uids, val_uids))
    return uid_splits