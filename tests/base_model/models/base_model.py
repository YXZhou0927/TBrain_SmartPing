"""
基礎模型類別，定義通用方法和XGBoost功能
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import time
import joblib
import os
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

class BaseMetaFeaturesGenerator(ABC):
    """Meta-Features生成器的基礎類別"""
    
    def __init__(self, model_type, use_gpu=True):
        """初始化
        
        Args:
            model_type (str): 模型類型
            use_gpu (bool): 是否使用GPU
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.all_models = {}
        self.all_scores = {}
        self.meta_features = {}
        self.mode_feature_idx = 0  # mode特徵的位置
    
    @abstractmethod
    def optimize_and_generate_meta(self, n_trials=100):
        """優化模型並生成meta-features
        
        Args:
            n_trials (int): Optuna trial數量
            
        Returns:
            pd.DataFrame: 訓練資料的meta-features
        """
        pass
    
    @abstractmethod
    def predict_test_data(self):
        """使用訓練好的模型預測測試資料
        
        Returns:
            tuple: 提交檔案和測試資料的meta-features
        """
        pass
    
    def calculate_auc(self, y_true, y_pred, n_classes):
        """計算AUC
        
        Args:
            y_true (np.array): 真實標籤
            y_pred (np.array): 預測機率
            n_classes (int): 類別數量
            
        Returns:
            float: AUC值
        """
        if n_classes == 2:
            # 二元分類
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                y_true_onehot[i, int(label)] = 1
            
            auc = roc_auc_score(y_true_onehot, y_pred, average='micro')
        else:
            # 多類別處理
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                y_true_onehot[i, int(label)] = 1
            
            auc = roc_auc_score(y_true_onehot, y_pred, average='micro', multi_class='ovr')
        
        return auc


class XGBoostBase(BaseMetaFeaturesGenerator):
    """XGBoost基礎模型類，實現通用的XGBoost功能"""
    
    def __init__(self, task_name, task_column, classes, cv_splits, scaler_type, model_type="xgb", use_gpu=True):
        """初始化通用XGBoost模型
        
        Args:
            task_name (str): 任務名稱
            task_column (str): 任務對應的列名
            classes (int): 類別數量
            cv_splits (int): 交叉驗證折數
            scaler_type (str): 標準化方法
            model_type (str): 模型類型
            use_gpu (bool): 是否使用GPU
        """
        super().__init__(model_type, use_gpu)
        self.task_name = task_name
        self.task_column = task_column
        self.classes = classes
        self.cv_splits = cv_splits
        self.scaler_type = scaler_type
        self.mode_feature_idx = 0
        self._config = self._load_config()
    
    def _load_config(self):
        """載入配置，子類應該覆寫此方法"""
        raise NotImplementedError("子類應覆寫此方法以提供正確的配置")
    
    def optimize_and_generate_meta(self, n_trials=None):
        """優化模型並生成meta-features
        
        Args:
            n_trials (int, optional): Optuna試驗次數
            
        Returns:
            pd.DataFrame: 訓練資料的meta-features
        """
        from config import base_config
        from utils import data_utils, file_utils
        
        if n_trials is None:
            n_trials = self._config.TRIAL_COUNT
            
        print(f"\n===== 開始處理 {self.task_name} 任務 =====")
        
        # 讀取訓練資料
        info = pd.read_csv(base_config.TRAIN_INFO_PATH)
        X, y_df, unique_ids, self.mode_feature_idx = data_utils.prepare_data(info, base_config.TRAIN_DATA_DIR)
        
        # 獲取任務標籤和玩家ID
        y_values = y_df[self.task_column].values
        player_ids = y_df['player_id'].values
        
        # 轉換標籤為從0開始的連續整數
        y = data_utils.transform_labels(y_values, self.task_name)
        
        # 創建DataFrame來記錄meta-features
        all_meta_features = pd.DataFrame({'unique_id': unique_ids})
        
        # 讀取訓練資料中的mode值
        train_modes = info.set_index('unique_id')['mode'].reindex(unique_ids).values
        all_meta_features['mode'] = train_modes
        
        # 模型名稱
        model_name = f"{self.model_type}_{self.task_name}"
        
        # 檢查是否已有優化結果
        study = file_utils.load_study(self.model_type, self.task_name, self._config.MODEL_DIR)
        
        if study is not None:
            print(f"載入已有的 {model_name} 優化結果")
            best_trial = study.best_trial
            best_models = best_trial.user_attrs["fold_models"]
            best_scores = best_trial.user_attrs["fold_scores"]
            oof_indices = best_trial.user_attrs["oof_indices"]
            oof_preds = best_trial.user_attrs["oof_predictions"]
            
            print(f"{model_name} 最佳AUC: {study.best_value:.4f}, 使用 {self.scaler_type} 標準化方法")
            
            # 保存結果
            self.all_models[self.task_name] = best_models
            self.all_scores[self.task_name] = best_scores
        else:
            print(f"為 {model_name} 開始超參數優化")
            
            # 創建優化研究
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
            sampler = optuna.samplers.TPESampler(n_startup_trials=10)
            study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
            
            # 進行優化
            study.optimize(
                lambda trial: self.xgb_objective(
                    trial, X, y, self.task_name, self.classes, 
                    self.scaler_type, player_ids, self.cv_splits, self.mode_feature_idx
                ),
                n_trials=n_trials
            )
            
            # 獲取最佳結果
            best_trial = study.best_trial
            best_models = best_trial.user_attrs["fold_models"]
            best_scores = best_trial.user_attrs["fold_scores"]
            best_params = best_trial.user_attrs["params"]
            oof_indices = best_trial.user_attrs["oof_indices"]
            oof_preds = best_trial.user_attrs["oof_predictions"]
            
            print(f"{model_name} 最佳AUC: {study.best_value:.4f}, 使用 {self.scaler_type} 標準化方法")
            
            # 保存結果
            self.all_models[self.task_name] = best_models
            self.all_scores[self.task_name] = best_scores
            
            # 保存study物件和參數
            file_utils.save_study(study, model_name, self._config.MODEL_DIR)
            file_utils.save_params(best_params, model_name, self._config.MODEL_DIR)
        
        # 保存最佳標準化器
        best_scaler = data_utils.get_scaler(self.scaler_type)
        if best_scaler is not None:
            # 只標準化非mode特徵
            _, features_to_scale = data_utils.split_features_by_mode(X, self.mode_feature_idx)
            best_scaler.fit(features_to_scale)
            file_utils.save_scaler(best_scaler, model_name, self.scaler_type, self._config.MODEL_DIR)
        
        # 生成meta-features - 這部分需要子類實現
        all_meta_features = self._generate_meta_features(all_meta_features, oof_indices, oof_preds, unique_ids)
        
        # 保存meta-features
        self.meta_features[self.task_name] = (oof_indices, oof_preds)
        
        # 保存模型資訊
        models = self.all_models[self.task_name]
        scores = self.all_scores[self.task_name]
        
        model_info = {
            'fold_scores': [float(score) for score in scores],
            'weights': [float(score)/sum(scores) for score in scores],  # 歸一化權重
            'mode_feature_idx': self.mode_feature_idx,
            'scaler_type': self.scaler_type
        }
        file_utils.save_model_info(model_info, model_name, self._config.MODEL_DIR)
        
        # 保存訓練資料的meta-features
        file_utils.save_meta_features(all_meta_features, self.model_type, self._config.META_FEATURES_DIR)
        
        # 保存CV結果
        self._save_cv_results(study)
        
        return all_meta_features
    
    def predict_test_data(self):
        """使用訓練好的模型預測測試資料
        
        Returns:
            tuple: 提交檔案和測試資料的meta-features
        """
        from config import base_config
        from utils import data_utils, file_utils
        
        # 讀取測試資料
        test_info = pd.read_csv(base_config.TEST_INFO_PATH)
        X_test, test_unique_ids, self.mode_feature_idx = data_utils.prepare_data(
            test_info, base_config.TEST_DATA_DIR, is_test=True
        )
        
        # 創建預測和meta-features DataFrame
        test_predictions = pd.DataFrame({'unique_id': test_unique_ids})
        test_meta_features = pd.DataFrame({'unique_id': test_unique_ids})
        
        # 模型名稱
        model_name = f"{self.model_type}_{self.task_name}"
        
        # 加載模型資訊
        model_info = file_utils.load_model_info(self.model_type, self.task_name, self._config.MODEL_DIR)
        
        if model_info is not None:
            weights = model_info['weights']
            self.mode_feature_idx = model_info.get('mode_feature_idx', 0)
            
            # 提取mode特徵和其他特徵
            mode_feature, features_to_scale = data_utils.split_features_by_mode(X_test, self.mode_feature_idx)
            
            # 應用標準化
            scaler = file_utils.load_scaler(self.model_type, self.task_name, self._config.MODEL_DIR)
            if scaler is not None:
                X_test_scaled = data_utils.apply_scaling(mode_feature, features_to_scale, scaler)
            else:
                X_test_scaled = X_test.copy()
            
            # 創建DMatrix用於預測
            categorical_features = [self.mode_feature_idx]
            feature_types = ['c' if i in categorical_features else 'q' for i in range(X_test_scaled.shape[1])]
            dtest = xgb.DMatrix(X_test_scaled, feature_types=feature_types)
            
            # 加載study以獲取模型
            study = file_utils.load_study(self.model_type, self.task_name, self._config.MODEL_DIR)
            if study is None:
                print(f"警告: 找不到{self.task_name}任務的study物件，跳過此任務")
                return None, None
                
            # 從study中獲取最佳模型
            best_trial = study.best_trial
            models = best_trial.user_attrs.get("fold_models", [])
            
            if not models:
                print(f"警告: 找不到{self.task_name}任務的模型，跳過此任務")
                return None, None
            
            print(f"已加載{len(models)}個{self.task_name}任務的模型")
            
            # 使用所有fold模型進行加權預測 - 這部分需要子類實現
            weighted_pred = self._predict_with_models(models, weights, dtest, len(X_test_scaled))
            
            # 將預測結果加入到測試meta-features和預測DataFrame中 - 這部分需要子類實現
            test_meta_features, test_predictions = self._process_predictions(test_meta_features, test_predictions, weighted_pred, test_unique_ids)
            
            # 保存測試資料的meta-features
            file_utils.save_meta_features(test_meta_features, self.model_type, self._config.META_FEATURES_DIR, is_test=True)
            
            # 格式化提交檔案
            sample_submission = pd.read_csv(base_config.SAMPLE_SUBMISSION_PATH)
            
            # 創建新的提交檔案，確保順序正確 - 這部分需要子類實現
            submission = self._format_submission(test_predictions, test_unique_ids, sample_submission)
            
            # 保存檔案
            file_utils.save_submission(submission, self.model_type, self._config.SUBMISSION_DIR)
            
            return submission, test_meta_features
        else:
            print(f"警告: 找不到{self.task_name}任務的模型資訊，跳過此任務")
            return None, None
    
    def xgb_objective(self, trial, X, y, task_name, n_classes, scaler_type, player_ids, n_splits, categorical_feature_idx=0):
        """XGBoost模型的Optuna優化目標函數
        
        Args:
            trial: Optuna trial物件
            X (np.array): 特徵
            y (np.array): 標籤
            task_name (str): 任務名稱
            n_classes (int): 類別數量
            scaler_type (str): 標準化方法
            player_ids (np.array): 玩家ID
            n_splits (int): 交叉驗證折數
            categorical_feature_idx (int): 類別特徵的索引
            
        Returns:
            float: 平均AUC
        """
        from utils import data_utils
        
        print(f"\n開始Trial {trial.number}的{task_name}任務優化...")
        start_time = time.time()
        
        # 取得標準化器
        scaler = data_utils.get_scaler(scaler_type)
        
        # 拆分mode特徵和其他特徵
        mode_feature, other_features = data_utils.split_features_by_mode(X, categorical_feature_idx)
        
        # 標準化並組合特徵
        if scaler is not None:
            scaled_features = scaler.fit_transform(other_features)
            X_scaled = np.column_stack((mode_feature, scaled_features))
        else:
            X_scaled = X.copy()
        
        # 定義XGBoost的超參數搜索空間
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "eta": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42
        }

        # 簡單的不平衡處理
        if n_classes == 2:
            # 二元分類 - 使用 scale_pos_weight
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 5.0)
        else:
            # 多類別分類 - 使用 max_delta_step
            params["max_delta_step"] = trial.suggest_int("max_delta_step", 0, 3)
        
        # 使用GPU
        if self.use_gpu:
            params["tree_method"] = "hist"  # 使用hist
            params["device"] = "cuda"       # 使用cuda
        
        # 為該任務準備分層交叉驗證
        unique_players = np.unique(player_ids)
        player_indices = {}
        player_labels = []
        
        # 獲取每個玩家的標籤和索引
        for i, player_id in enumerate(unique_players):
            player_mask = player_ids == player_id
            player_indices[player_id] = np.where(player_mask)[0]
            # 將該玩家的第一個標籤作為代表
            player_labels.append(y[player_mask][0])
        
        # 使用任務特定的折數進行交叉驗證
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(unique_players, player_labels))
        
        # 每輪迭代次數
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        
        # 根據任務設置目標函數
        if n_classes == 2:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
        else:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
        
        # 處理所有fold
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            result = self._process_fold(fold, train_idx, val_idx, unique_players, player_indices, 
                                       X_scaled, y, n_classes, params, n_estimators, 
                                       n_splits, task_name, categorical_feature_idx, trial)
            fold_results.append(result)
        
        # 整理結果
        fold_models = [res['model'] for res in fold_results]
        fold_scores = [res['auc'] for res in fold_results]
        
        # 整合out-of-fold預測
        all_indices = np.concatenate([res['val_idx'] for res in fold_results])
        all_preds = np.vstack([res['y_pred'] for res in fold_results])
        
        # 確保索引順序正確
        sort_indices = np.argsort(all_indices)
        ordered_indices = all_indices[sort_indices]
        ordered_preds = all_preds[sort_indices]
        
        # 保存這個trial的信息
        trial.set_user_attr("fold_models", fold_models)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("params", params)
        trial.set_user_attr("oof_indices", ordered_indices)
        trial.set_user_attr("oof_predictions", ordered_preds)
        trial.set_user_attr("scaler_type", scaler_type)
        trial.set_user_attr("n_estimators", n_estimators)
        
        mean_auc = np.mean(fold_scores)
        total_time = time.time() - start_time
        print(f"Trial {trial.number}的{task_name}任務完成，平均AUC: {mean_auc:.4f}，總耗時: {total_time:.2f}秒")
        
        return mean_auc
    
    def _process_fold(self, fold, train_player_idx, val_player_idx, unique_players, player_indices, 
                     X_scaled, y, n_classes, params, n_estimators, n_splits, task_name, categorical_feature_idx, trial):
        """處理單個交叉驗證fold
        
        Args:
            fold (int): fold索引
            train_player_idx (np.array): 訓練集玩家索引
            val_player_idx (np.array): 驗證集玩家索引
            unique_players (np.array): 唯一玩家ID
            player_indices (dict): 每個玩家的資料索引
            X_scaled (np.array): 標準化後的特徵
            y (np.array): 標籤
            n_classes (int): 類別數量
            params (dict): XGBoost參數
            n_estimators (int): 迭代次數
            n_splits (int): 交叉驗證折數
            task_name (str): 任務名稱
            categorical_feature_idx (int): 類別特徵索引
            trial: Optuna trial物件
            
        Returns:
            dict: 處理結果
        """
        print(f"  處理{task_name}的第{fold+1}/{n_splits}折...")
        fold_start_time = time.time()
        
        train_players = unique_players[train_player_idx]
        val_players = unique_players[val_player_idx]
        
        # 收集訓練集和驗證集索引
        train_idx = []
        for player_id in train_players:
            train_idx.extend(player_indices[player_id])
        train_idx = np.array(train_idx)
        
        val_idx = []
        for player_id in val_players:
            val_idx.extend(player_indices[player_id])
        val_idx = np.array(val_idx)
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 指定mode特徵為類別特徵
        categorical_features = [categorical_feature_idx]
        feature_types = ['c' if i in categorical_features else 'q' for i in range(X_train.shape[1])]
        
        # 創建DMatrix並處理樣本權重
        if n_classes > 2:
            # 計算類別頻率和樣本權重
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            total_samples = len(y_train)
            
            # 計算較為溫和的樣本權重
            sample_weights = np.ones(len(y_train))
            for cls, count in zip(unique_classes, class_counts):
                ratio = total_samples / count
                weight = np.sqrt(ratio)  # 平方根可以減緩極端影響
                sample_weights[y_train == cls] = weight
            
            # 創建帶權重的DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, feature_types=feature_types)
        else:
            # 二元分類主要依賴scale_pos_weight
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_types=feature_types)
            
        dval = xgb.DMatrix(X_val, label=y_val, feature_types=feature_types)
        
        # 訓練設置
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        # 訓練模型
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=n_estimators, 
            evals=watchlist, 
            early_stopping_rounds=20, 
            verbose_eval=False
        )
        
        # 預測驗證集
        val_predictions = model.predict(dval)
        
        # 調整輸出格式
        if n_classes > 2:
            # 轉換為每個類別的機率矩陣
            val_predictions = val_predictions.reshape(len(y_val), n_classes)
        else:
            # 二元分類，轉換為兩列矩陣 [p, 1-p]
            val_predictions = np.vstack((1 - val_predictions, val_predictions)).T
        
        # 計算AUC
        auc = self.calculate_auc(y_val, val_predictions, n_classes)
        
        fold_time = time.time() - fold_start_time
        print(f"  {task_name}第{fold+1}/{n_splits}折完成，AUC: {auc:.4f}，耗時: {fold_time:.2f}秒")
        
        # 報告中間結果以支持提前停止
        trial.report(auc, fold)
        
        # 如果性能不佳，可能會提前停止
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return {
            'model': model,
            'val_idx': val_idx,
            'y_pred': val_predictions,
            'auc': auc
        }
    
    def _save_cv_results(self, study=None):
        """保存CV結果
        
        Args:
            study: Optuna study物件
        """
        if self.task_name not in self.all_scores or self.task_name not in self.all_models:
            print(f"警告: 找不到{self.task_name}任務的CV結果")
            return
            
        # 計算基本指標
        fold_scores = self.all_scores[self.task_name]
        mean_auc = float(np.mean(fold_scores))
        
        # 獲取該任務的最佳trial
        best_params = {}
        if study:
            best_trial = study.best_trial
            best_params = best_trial.user_attrs.get('params', {}).copy()
        
        # 整合到結果中
        cv_results = {
            'task_name': self.task_name,
            'best_auc': mean_auc,
            'best_scaler': self.scaler_type,
            'best_params': best_params,
            'fold_summary': [float(score) for score in fold_scores]
        }
        
        # 保存CV結果
        cv_results_path = os.path.join(self._config.MODEL_DIR, f"{self.model_type}_{self.task_name}_cv_results.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        print(f"CV結果已保存至: {cv_results_path}")
    
    # 以下方法需要由子類實現
    
    def _generate_meta_features(self, all_meta_features, oof_indices, oof_preds, unique_ids):
        """生成meta-features
        
        Args:
            all_meta_features (pd.DataFrame): meta-features DataFrame
            oof_indices (np.array): out-of-fold索引
            oof_preds (np.array): out-of-fold預測
            unique_ids (list): 唯一ID列表
            
        Returns:
            pd.DataFrame: 更新後的meta-features DataFrame
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def _predict_with_models(self, models, weights, dtest, n_samples):
        """使用模型進行預測
        
        Args:
            models (list): 模型列表
            weights (list): 權重列表
            dtest: XGBoost DMatrix
            n_samples (int): 樣本數量
            
        Returns:
            np.array: 預測結果
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def _process_predictions(self, test_meta_features, test_predictions, weighted_pred, test_unique_ids):
        """處理預測結果
        
        Args:
            test_meta_features (pd.DataFrame): 測試meta-features
            test_predictions (pd.DataFrame): 測試預測
            weighted_pred (np.array): 加權預測結果
            test_unique_ids (list): 測試唯一ID
            
        Returns:
            tuple: 更新後的test_meta_features和test_predictions
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def _format_submission(self, test_predictions, test_unique_ids, sample_submission):
        """格式化提交文件
        
        Args:
            test_predictions (pd.DataFrame): 測試預測
            test_unique_ids (list): 測試唯一ID
            sample_submission (pd.DataFrame): 樣本提交文件
            
        Returns:
            pd.DataFrame: 格式化後的提交文件
        """
        raise NotImplementedError("子類必須實現此方法")