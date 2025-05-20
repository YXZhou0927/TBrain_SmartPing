"""
打球年限任務的XGBoost模型
"""

import numpy as np
import pandas as pd
from models.base_model import XGBoostBase

class PlayYearsXGBGenerator(XGBoostBase):
    """打球年限任務的XGBoost特徵生成器"""
    
    def __init__(self, use_gpu=True):
        """初始化
        
        Args:
            use_gpu (bool): 是否使用GPU
        """
        from config import play_years_config as config
        super().__init__(
            task_name=config.TASK_NAME,
            task_column=config.TASK_COLUMN,
            classes=config.CLASSES,
            cv_splits=config.CV_SPLITS,
            scaler_type=config.SCALER_TYPE,
            model_type="xgb", 
            use_gpu=use_gpu
        )
    
    def _load_config(self):
        """載入配置"""
        from config import play_years_config
        return play_years_config
    
    def _generate_meta_features(self, all_meta_features, oof_indices, oof_preds, unique_ids):
        """生成打球年限任務的meta-features
        
        Args:
            all_meta_features (pd.DataFrame): meta-features DataFrame
            oof_indices (np.array): out-of-fold索引
            oof_preds (np.array): out-of-fold預測
            unique_ids (list): 唯一ID列表
            
        Returns:
            pd.DataFrame: 更新後的meta-features DataFrame
        """
        # 多類別任務保存所有類別的機率
        for i in range(self.classes):
            col_name = f"{self.task_name}_{i}"
            meta_values = np.zeros(len(unique_ids))
            meta_values[oof_indices] = oof_preds[:, i]
            all_meta_features[col_name] = meta_values
        return all_meta_features
    
    def _predict_with_models(self, models, weights, dtest, n_samples):
        """使用模型進行打球年限預測
        
        Args:
            models (list): 模型列表
            weights (list): 權重列表
            dtest: XGBoost DMatrix
            n_samples (int): 樣本數量
            
        Returns:
            np.array: 預測結果
        """
        # 多類別分類
        weighted_pred = np.zeros((n_samples, self.classes))
        for i, model in enumerate(models):
            if i < len(weights):
                # 多類別預測已經是矩陣形式
                pred = model.predict(dtest).reshape(n_samples, self.classes)
                weighted_pred += weights[i] * pred
        return weighted_pred
    
    def _process_predictions(self, test_meta_features, test_predictions, weighted_pred, test_unique_ids):
        """處理打球年限預測結果
        
        Args:
            test_meta_features (pd.DataFrame): 測試meta-features
            test_predictions (pd.DataFrame): 測試預測
            weighted_pred (np.array): 加權預測結果
            test_unique_ids (list): 測試唯一ID
            
        Returns:
            tuple: 更新後的test_meta_features和test_predictions
        """
        # 將預測結果加入到測試meta-features中
        for i in range(self.classes):
            col_name = f"{self.task_name}_{i}"
            test_meta_features[col_name] = weighted_pred[:, i]
            
            # 保存到預測DataFrame - 使用原始提交文件的格式
            test_predictions[f"{self.task_column}_{i}"] = weighted_pred[:, i]
        
        return test_meta_features, test_predictions
    
    def _format_submission(self, test_predictions, test_unique_ids, sample_submission):
        """格式化打球年限任務的提交文件
        
        Args:
            test_predictions (pd.DataFrame): 測試預測
            test_unique_ids (list): 測試唯一ID
            sample_submission (pd.DataFrame): 樣本提交文件
            
        Returns:
            pd.DataFrame: 格式化後的提交文件
        """
        submission = pd.DataFrame({'unique_id': test_unique_ids})
        
        # 添加每個類別的機率
        for i in range(self.classes):
            col_name = f"{self.task_column}_{i}"
            if col_name in test_predictions.columns:
                submission[col_name] = test_predictions[col_name].round(4)
        
        submission = submission.set_index('unique_id').loc[sample_submission['unique_id']].reset_index()
        return submission