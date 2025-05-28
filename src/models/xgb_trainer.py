from .base_trainer import ModelTrainer
from ..config import XGBTrainerConfig
import xgboost as xgb
import numpy as np
import pandas as pd
import optuna
from typing import Optional, Tuple, Union

class XGBTrainer(ModelTrainer):
    def __init__(self, task_name: str, use_gpu: bool = True):
        """
        初始化XGBoost訓練器
        Args:
            task_name (str): 任務名稱
            use_gpu (bool): 是否使用GPU進行訓練，默認為True
        """
        super().__init__(task_name=task_name, use_gpu=use_gpu)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.trainer_config = XGBTrainerConfig()
        self.random_state = self.trainer_config.get_random_state()
        self.model_dir = self.trainer_config.get_model_dir()
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        

    def set_train(self, train_df: pd.DataFrame):
        target_col = self.target_col
        scaler_type = self.scaler_type
        self.train_data = self._set_data(train_df, target_col, scaler_type=scaler_type)
    def set_val(self, val_df: pd.DataFrame):
        target_col = self.target_col
        scaler_type = self.scaler_type
        self.val_data = self._set_data(val_df, target_col, scaler_type=scaler_type)
    def set_test(self, test_df: pd.DataFrame):
        scaler_type = self.scaler_type
        self.test_data = self._set_data(test_df, target_col=None, scaler_type=scaler_type)

    def train(self, xgb_params: dict, n_estimators: int, early_stopping_rounds: int = 20, verbose_eval: bool = False, 
              evaluate_method: str = 'auc', model_name: Optional[str] = None, save_model: bool = True) -> xgb.Booster:
        """
        訓練XGBoost模型
        Args:
            xgb_params (dict): XGBoost模型參數
            n_estimators (int): 迭代次數
            early_stopping_rounds (int): 早停輪次
            verbose (bool): 是否輸出訓練過程
            evaluate_method (str): 評估方法，默認為 'auc'
            model_name (Optional[str]): 模型名稱，如果為None則自動生成
            save_model (bool): 是否保存模型到文件，默認為True
        Returns:
            model: 訓練好的XGBoost模型
        """
        if not self.train_data:
            raise ValueError("Training data is not set. Please call set_train() first.")
        if not xgb_params:
            raise ValueError("No parameters provided for training. Please provide XGBoost parameters.")
        
        dtrain = self.train_data
        dval = self.val_data if self.val_data else None
        watchlist = [(dtrain, 'train')]
        if dval is not None:
            watchlist.append((dval, 'eval'))

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        if model_name is None:
            import time
            model_name = f"XGBTrainer_{time.strftime('%Y%m%d_%H%M%S')}"

        if dval is not None:
            # Get y_val from dval
            y_val = dval.get_label()
            y_val = np.asarray(y_val, dtype=int)
            y_val_onehot = np.eye(self.n_classes)[y_val]  # one-hot 編碼
            val_preds = model.predict(dval).reshape(-1, self.n_classes) 
            score = self.evaluate(y_val_onehot, val_preds, method=evaluate_method)
            self.models[model_name] = {
                "class_name": "XGBTrainer", 
                "model": model, 
                "eval_method": evaluate_method, 
                "score": score
            }
        else:
            self.models[model_name] = {
                "class_name": "XGBTrainer", 
                "model": model, 
                "eval_method": None, 
                "score": None
            }
        
        if save_model:
            model_path = self.model_dir / f"{model_name}.pkl"
            self.save_model(model_name, model_path)
            print(f"Model {model_name} trained and saved to {model_path}")
            
        return model

    def auto_optimize(self, info_df: pd.DataFrame):
        """
        使用Optuna自動優化XGBoost模型參數
        Args:
            info_df (pd.DataFrame): 包含特徵和目標值的DataFrame
        Returns:
            None
        """
        if not isinstance(info_df, pd.DataFrame):
            raise ValueError("info_df must be a pandas DataFrame.")
        if info_df.empty:
            raise ValueError("info_df is empty. Please provide a DataFrame with data.")
        if self.target_col not in info_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in info_df. Please provide a valid DataFrame.")
        
        # 取得 Hyperparameter調整的折數
        n_folds = self.trainer_config.get_cv_folds()[self.target_col]
        n_trials = self.trainer_config.get_n_trials()
        scoring = self.trainer_config.get_scoring()
        
        # 放入調參邏輯 (GridSearchCV / Optuna 等)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
        sampler = optuna.samplers.TPESampler(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
        study.optimize(lambda trial: self._objective(trial, info_df, n_folds=n_folds, eval_method=scoring), n_trials=n_trials)
        print(f"Best trial: {study.best_trial.number} with score: {study.best_trial.value}")
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S") # timestamp for model name
        filepath = self.model_dir / f"{timestamp}_{self.task_name}_xgb_study.pkl"
        self._save_study(study, str(filepath))


    def predict(self, model_name: str, test_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        使用指定的模型進行預測
        Args:
            model_name (str): 要使用的模型名稱
            test_df (pd.DataFrame): 包含測試數據的DataFrame，未提供則自動載入 self.test_data
            scaler_type (str): 設定數據前處理的方法，未提供則使用
        Returns:
            pd.DataFrame: 預測結果的DataFrame
        """
        if not self.models:
            raise ValueError("No trained models available. Please train a model first.")
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in trained models. Available models: {self.show_models()}")
        if test_df is None and not self.test_data:
            raise ValueError("Test data is not set. Please call set_test() first or provide test_df.")
        if test_df is not None:
            self.set_test(test_df=test_df)
        
        model_info = self.models[model_name]
        model = model_info['model']
        if not isinstance(model, xgb.Booster):
            raise ValueError(f"Model '{model_name}' is not a valid XGBoost model.")
        
        return self._process_predictions(model.predict(self.test_data))
        
    
    def _set_data(self, info_df: pd.DataFrame, target_col: str = None, scaler_type = 'minmax') -> xgb.DMatrix:
        """
        設置訓練、驗證或測試數據
        Args:
            info_df (pd.DataFrame): 包含特徵和目標值的DataFrame
            target_col (str): 目標列名，如果為None則不使用目標值
            scaler_type (str): 特徵標準化方法，默認為 'minmax'
        Returns:
            xgb.DMatrix: XGBoost的DMatrix格式數據
        """
        uid, X_df, y_df = self._process_info_df(info_df, target_col)
        X_scaled = self._process_features(X_df, scaler_type=scaler_type)
        X_scaled = np.column_stack((uid['mode'].values, X_scaled))
        X_types = ['c'] + ['q'] * (X_scaled.shape[1] - 1) # mode is categorical and other features are numerical
        if y_df is not None:
            if y_df.shape[1] != 1:
                raise ValueError("y_df should contain only one column for the target variable.")
            y_df = self.data_config.get_label_encoded_targets(y_df)
            y_series = y_df.squeeze()
            # Calculate ratio of each class
            class_counts = y_df.value_counts(normalize=True)
            sample_weight = np.ones(len(y_df))
            if len(class_counts) > 1:   
                # If there are multiple classes, calculate sample weights based on class ratio
                for cls, ratio in class_counts.items():
                    sample_weight[y_series == cls] = np.sqrt(1 / ratio) # to balance the classes

            return xgb.DMatrix(X_scaled, label=y_series, weight=sample_weight, feature_types=X_types)
        else:
            return xgb.DMatrix(X_scaled, feature_types=X_types)
 
    def _objective(self, trial: optuna.Trial, info_df: pd.DataFrame, n_folds: int, eval_method: str = 'auc') -> float:
        """
        Objective function for Optuna optimization
        Args:
            trial (optuna.Trial): Optuna trial object
            info_df (pd.DataFrame): DataFrame containing features and target values
            n_folds (int): Number of folds for cross-validation
            eval_method (str): Evaluation method, default is 'auc'
        Returns:
            float: Objective value to be maximized (e.g., auc score)
        """
        parameter_space = self.trainer_config.get_parameter_space()
        xgb_params = {
            "objective": parameter_space["objective"],
            "eval_metric": parameter_space["eval_metric"],
            "num_class": parameter_space["num_class"][self.target_col],
            "max_depth": trial.suggest_int("max_depth", parameter_space["max_depth"][0], parameter_space["max_depth"][-1]),
            "eta": trial.suggest_float("learning_rate", parameter_space["learning_rate"][0], parameter_space["learning_rate"][-1]),
            "subsample": trial.suggest_float("subsample", parameter_space["subsample"][0], parameter_space["subsample"][-1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", parameter_space["colsample_bytree"][0], parameter_space["colsample_bytree"][-1]),
            "min_child_weight": trial.suggest_int("min_child_weight", parameter_space["min_child_weight"][0], parameter_space["min_child_weight"][-1]),
            "gamma": trial.suggest_float("gamma", parameter_space["min_child_weight"][0], parameter_space["min_child_weight"][-1]),
            "random_state": self.random_state,
        }
        if self.n_classes > 2:
            xgb_params['max_delta_step'] = trial.suggest_int("max_delta_step", parameter_space["max_delta_step"][0], parameter_space["max_delta_step"][-1])
        if parameter_space["use_gpu"]:
            xgb_params["tree_method"] = "hist"  # 使用hist
            xgb_params["device"] = "cuda"
        n_estimators = trial.suggest_int("n_estimators", parameter_space["n_estimators"][0], parameter_space["n_estimators"][-1])

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S") # timestamp for model name
        uid_splits = self._split_by_player_id(info_df, n_splits=n_folds)
        model_list = []
        score_list = []
        for fold, (train_uids, val_uids) in enumerate(uid_splits):
            print(f"  處理{self.task_name}的第{fold+1}/{n_folds}折...")
            fold_start_time = time.time()

            self.set_train(info_df[info_df['unique_id'].isin(train_uids)])
            self.set_val(info_df[info_df['unique_id'].isin(val_uids)])
            model_name = f"{timestamp}_Task_{self.task_name}_XGBTrial{trial.number}_Fold{fold+1}"
            self.train(xgb_params, n_estimators=n_estimators, early_stopping_rounds=20, verbose_eval=False,
                               evaluate_method=eval_method, model_name=model_name, save_model=True)
            fold_score = self.models[model_name]['score']  # Get the score from the model info
            model_list.append(model_name)
            score_list.append(fold_score)

            fold_time = time.time() - fold_start_time
            print(f"  {self.task_name}第{fold+1}/{n_folds}折完成，{eval_method}: {fold_score:.4f}，耗時: {fold_time:.2f}秒")

            trial.report(fold_score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = np.mean(score_list)
        print(f"  {self.task_name}所有折的平均{eval_method}: {score:.4f}")

        # Set user attributes for the trial
        trial.set_user_attr("task_name", self.task_name)
        trial.set_user_attr("n_folds", n_folds)
        trial.set_user_attr("eval_method", eval_method)
        trial.set_user_attr("uid_splits", uid_splits)
        trial.set_user_attr("model_list", model_list)
        trial.set_user_attr("score_list", score_list)
        trial.set_user_attr("xgb_params", xgb_params)
        trial.set_user_attr("n_estimators", n_estimators)
        return score
    
    def _save_study(self, study: optuna.Study, file_path: str):
        """
        Save the Optuna study to a file
        Args:
            study (optuna.Study): The Optuna study to save
            file_path (str): Path to save the study
        """
        import joblib
        joblib.dump(study, file_path)
        print(f"Optuna study saved to {file_path}")