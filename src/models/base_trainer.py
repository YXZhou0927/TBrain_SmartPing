import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Union
from ..config import DataConfigManager

class ModelTrainer:
    def __init__(self, task_name: str, use_gpu: bool = True):
        self.use_gpu = use_gpu  # Whether to use GPU for training
        self.data_config = DataConfigManager()
        self.target_col = self.task_name = task_name
        self.one_hot_columns = self.data_config.get_columns(self.task_name)  # Get one-hot encoding columns for the task
        self.n_classes = len(self.one_hot_columns)  # Number of classes for the task
        self.models = {}  # {model_name: {class_name: str, model, eval_method: str, score: float}}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler_type = self.data_config.get_scaler_type(self.target_col)
        
    def set_train(self, train_df: pd.DataFrame):
        raise NotImplementedError
    
    def set_val(self, val_df: pd.DataFrame):
        raise NotImplementedError
    
    def set_test(self, test_df: pd.DataFrame):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def load_model(self, model_name: str, filename: Union[str, Path]):
        """
        Load a model from a file and add it to the models dictionary.
        Args:
            model_name (str): The name of the model to be loaded.
            filename (str | Path): The path to the model file.
        """
        from .model_registry import load_model_file
        if not isinstance(filename, (str, Path)):
            raise ValueError("Filename must be a string or Path object.")
        if isinstance(filename, Path):
            filename = str(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} does not exist.")
        model_data = load_model_file(filename)
        self.models[model_name] = model_data

    def save_model(self, model_name: str, filename: Union[str, Path]):
        """
        Save a model to a file.
        Args:
            model_name (str): The name of the model to be saved.
            filename (str | Path): The path where the model will be saved.
        """
        from .model_registry import save_model_file
        if not isinstance(filename, (str, Path)):
            raise ValueError("Filename must be a string or Path object.")
        if isinstance(filename, Path):
            filename = str(filename)
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in the models dictionary.")
        save_model_file(self.models[model_name], filename)

    def copy_model(self, model_name: str, new_model_name: str):
        """
        Copy a model from one name to another in the models dictionary.
        Args:
            model_name (str): The name of the model to be copied.
            new_model_name (str): The new name for the copied model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in the models dictionary.")
        self.models[new_model_name] = self.models[model_name]

    def delete_model(self, model_name: str):
        """
        Delete a model from the models dictionary.
        Args:
            model_name (str): The name of the model to be deleted.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in the models dictionary.")
        del self.models[model_name]

    def show_models(self):
        return list(self.models.keys())

    def evaluate(self, true_onehot: np.ndarray, preds: np.ndarray, method: str = "auc"):
        from .evaluate import evaluate_model
        return evaluate_model(true_onehot, preds, method)

    def _process_info_df(self, info_df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.DataFrame, None]]:
        """
        處理信息DataFrame，提取特徵和目標列
        Args:
            info_df (pd.DataFrame): 包含信息的DataFrame
            target_col (str): 目標列名，默認為None，表示不提取目標列
        Returns:
            tuple: 包含處理後的DataFrame、特徵和目標列（若未提供目標列則返回None）
        """
        from .process import process_info_df
        return process_info_df(info_df=info_df, target_col=target_col)

    def _process_predictions(self, preds: np.ndarray):
        """
        處理預測結果NumpyArray，合併成DataFrame
        Args:
            preds (np.ndarray): 包含預測結果的NumpyArray
        Returns:
            pd.DataFrame: 包含預測結果和輸出欄位的DataFrame
        """
        from .process import process_predictions
        return process_predictions(preds, self.one_hot_columns)
    
    def _process_features(self, features: Union[pd.DataFrame, np.ndarray], scaler_type='minmax'):
        """
        處理特徵DataFrame，將mode特徵與其他特徵分開，並應用標準化。
        Args:
            features (pd.DataFrame | np.ndarray): 包含特徵的DataFrame或np.ndarray。
            scaler_type (str): 標準化方法，默認為 'minmax'。
        Returns:
            np.array: 處理後的特徵數組
        """
        from .process import process_features
        return process_features(features, scaler_type)

    def _split_by_player_id(self, info_df: pd.DataFrame, n_splits: int = 5) -> list:
        """
        使用GroupKFold按player_id分割數據，返回訓練和驗證唯一ID的分割列表。
        Args:
            info_df (pd.DataFrame): 包含特徵和目標值的DataFrame
            n_splits (int): 分割數量，默認為5
        Returns:
            list: 包含訓練和驗證唯一ID的分割列表
        """
        from .utils.cross_validate import group_kfold_by_player_id
        return group_kfold_by_player_id(info_df, self.target_col, n_splits, self.random_state)