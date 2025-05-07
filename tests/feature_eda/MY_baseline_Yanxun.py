import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path 
import warnings
warnings.filterwarnings('ignore')

class BadmintonPredictor:
    def __init__(self):
        self.all_models = {}  # 保存所有CV模型
        self.all_scores = {}  # 保存所有CV分數
        self.scaler = MinMaxScaler()
        self.cv_splits_map = {
            'gender': 8,
            'hold_racket_handed': 5,
            'play_years': 5,
            'level': 3
        }
    
    def prepare_data(self, info, datapath):
        """統一的資料準備函數"""
        datalist = list(Path(datapath).glob('**/*.csv'))
        X_data = []
        y_data = []
        unique_ids = []
        player_ids = []
        
        for file in datalist:
            unique_id = int(Path(file).stem)
            
            # 如果是訓練資料，需要檢查是否在info中
            if not info.empty:
                row = info[info['unique_id'] == unique_id]
                if row.empty:
                    continue
            
            data = pd.read_csv(file)
            # if len(data) != 27:
            #     print(f"Warning: {file} has {len(data)} rows instead of 27")
            #     continue
            
            # 將27x34的資料展平成1x918的特徵向量
            features = data.values.flatten()
            
            if not info.empty:  # 訓練資料
                labels = {
                    'gender': row['gender'].iloc[0],
                    'hold_racket_handed': row['hold racket handed'].iloc[0],
                    'play_years': row['play years'].iloc[0],
                    'level': row['level'].iloc[0]
                }
                y_data.append(labels)
                player_ids.append(row['player_id'].iloc[0])
            
            X_data.append(features)
            unique_ids.append(unique_id)
        
        X = np.array(X_data)
        if not info.empty:
            y_df = pd.DataFrame(y_data, index=unique_ids)
            y_df['player_id'] = player_ids
            return X, y_df
        else:
            return X, unique_ids
    
    def get_predictions(self, model, X):
        """統一的預測函數 - 現在所有任務都返回所有類別的機率"""
        return model.predict_proba(X)
    
    def calculate_auc(self, y_true, y_pred, task):
        """統一的AUC計算函數"""
        if task in ['gender', 'hold_racket_handed']:
            # 二元分類 - 使用一致的方法處理
            # 建立 one-hot 編碼
            n_classes = 2
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                y_true_onehot[i, int(label)-1] = 1
            
            # 計算 AUC
            auc = roc_auc_score(y_true_onehot, y_pred, average='micro', multi_class='ovr')
        else:
            # 多類別處理
            if task == 'play_years':
                n_classes = 3
                y_true_onehot = np.zeros((len(y_true), n_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, int(label)] = 1
            else:  # level
                n_classes = 4
                y_true_onehot = np.zeros((len(y_true), n_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, int(label)-2] = 1
            
            auc = roc_auc_score(y_true_onehot, y_pred, average='micro', multi_class='ovr')
        
        return auc
    
    def cross_validate(self):
        """執行交叉驗證並保存所有模型"""
        train_info_path = Path('data/raw/39_Training_Dataset/train_info.csv')
        data_dir = Path('data/processed/TimeSeriesFeatures/TrainingData/features')
        info = pd.read_csv(str(train_info_path))
        X, y_df = self.prepare_data(info, str(data_dir))
        X_scaled = self.scaler.fit_transform(X)
        
        task_scores = {}
        
        for task in ['gender', 'hold_racket_handed', 'play_years', 'level']:
            print(f"\nTraining for task: {task}")
            n_splits = self.cv_splits_map[task]
            
            # 分層抽樣
            unique_players = y_df['player_id'].unique()
            player_labels = []
            for player_id in unique_players:
                player_rows = y_df[y_df['player_id'] == player_id]
                player_labels.append(player_rows[task].iloc[0])
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_scores = []
            fold_models = []
            
            for fold, (train_players_idx, val_players_idx) in enumerate(skf.split(unique_players, player_labels)):
                train_players = unique_players[train_players_idx]
                val_players = unique_players[val_players_idx]
                
                train_mask = y_df['player_id'].isin(train_players)
                val_mask = y_df['player_id'].isin(val_players)
                
                train_idx = np.where(train_mask)[0]
                val_idx = np.where(val_mask)[0]
                
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = y_df[task].iloc[train_idx].values
                y_val = y_df[task].iloc[val_idx].values
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # 直接使用原始標籤，不進行變換
                model.fit(X_train, y_train)
                
                # 獲取所有類別的預測機率
                val_predictions = self.get_predictions(model, X_val)
                
                # 計算 AUC
                auc = self.calculate_auc(y_val, val_predictions, task)
                
                fold_scores.append(auc)
                fold_models.append(model)
                print(f"  Fold {fold + 1}/{n_splits} - AUC: {auc:.4f}")
            
            avg_score = np.mean(fold_scores)
            task_scores[task] = avg_score
            print(f"  Average AUC for {task}: {avg_score:.4f}")
            
            # 保存所有模型和分數
            self.all_models[task] = fold_models
            self.all_scores[task] = fold_scores
        
        final_score = np.mean(list(task_scores.values()))
        print(f"\nFinal Cross-Validation Score: {final_score:.4f}")
        
        return task_scores
    
    def generate_submission(self):
        """生成提交檔案，使用所有CV模型的加權集成"""
        # 測試資料
        test_info_path = Path('data/raw/39_Test_Dataset/test_info.csv')
        test_data_dir = Path('data/processed/TimeSeriesFeatures/TestingData/features')
        test_info = pd.read_csv(str(test_info_path))
        X_test, test_unique_ids = self.prepare_data(pd.DataFrame(), str(test_data_dir))
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = {}
        
        # 對每個任務進行加權集成預測
        for task in ['gender', 'hold_racket_handed', 'play_years', 'level']:
            models = self.all_models[task]
            scores = self.all_scores[task]
            
            # 為每個模型計算權重（基於AUC）
            weights = np.array(scores)
            weights = weights / weights.sum()  # 歸一化
            
            # 收集每個模型的預測
            all_preds = []
            for i, model in enumerate(models):
                pred = self.get_predictions(model, X_test_scaled)
                all_preds.append(pred)
            
            # 加權平均 - 對每個類別進行加權平均
            # 假設第一個模型的預測形狀為參考
            n_samples, n_classes = all_preds[0].shape
            weighted_pred = np.zeros((n_samples, n_classes))
            
            for i, pred in enumerate(all_preds):
                weighted_pred += weights[i] * pred
            
            predictions[task] = weighted_pred
        
        # 創建提交檔案
        submission = pd.DataFrame({'unique_id': test_unique_ids})
        
        # 針對每個任務選擇正確的輸出
        # 性別：取第一個類別的機率 (男性=1的機率)
        submission['gender'] = predictions['gender'][:, 0]
        
        # 持拍手：取第一個類別的機率 (右手=1的機率)
        submission['hold racket handed'] = predictions['hold_racket_handed'][:, 0]
        
        # 打球年資：取每個類別的機率
        for i in range(3):
            submission[f'play years_{i}'] = predictions['play_years'][:, i]
        
        # 水平級別：取每個類別的機率
        for level in [2, 3, 4, 5]:
            submission[f'level_{level}'] = predictions['level'][:, level-2]
        
        # 確保順序正確
        sample_submission_path = Path('data/raw/39_Test_Dataset/sample_submission.csv')
        sample_submission = pd.read_csv(str(sample_submission_path))
        submission = submission.set_index('unique_id').loc[sample_submission['unique_id']].reset_index()
        
        # 保存檔案
        submission_path = Path('outputs/submissions')
        current_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        submission_path.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path / f'submission_{current_time}.csv', index=False)
        print("\nSubmission file created: submission.csv")
        return submission

if __name__ == "__main__":
    predictor = BadmintonPredictor()
    
    # 執行交叉驗證
    cv_scores = predictor.cross_validate()
    
    # 生成提交檔案（使用所有CV模型的加權集成）
    submission = predictor.generate_submission()