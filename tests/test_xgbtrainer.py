"""
測試模組：TestXGBTrainer

此模組負責對 `XGBTrainer` 類別與其訓練、預測、儲存、載入等功能進行單元測試。

使用資料來源：
- 若 `tests/train_data.csv` 或 `tests/test_data.csv` 存在，將使用快取資料以節省測試時間。
- 否則會從 `data/raw/...` 和 `data/processed/...` 中合併生成訓練/測試資料並快取。

測試項目說明：
- test_xgb_trainer:
    測試 XGBTrainer 類別是否正確初始化。

- test_train:
    測試 auto_optimize 是否能正確訓練模型，並且模型儲存於 trainer.models。

- test_predict:
    測試是否能設定測試資料並執行預測，預期會回傳一個預測 DataFrame。

- test_load_model:
    預留：將測試模型是否能成功載入，並儲存至 models dict 中。

- test_save_model:
    預留：將測試模型是否能正確儲存至指定路徑。

- test_copy_model:
    預留：將測試是否能從現有模型建立副本，並註冊為新模型名稱。

- test_set_train / test_set_val / test_set_test:
    預留：將測試是否能成功設定訓練、驗證、測試資料集。

使用方式：
    python -m unittest tests/test_xgbtrainer.py

建議整合至 CI 工具時先快取 `train_data.csv` 減少依賴實體檔案與運算負擔。
"""
import unittest
from pathlib import Path
import pandas as pd
import sys

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent  # Assuming the script is in tests directory
sys.path.append(str(root_dir))

from src import XGBTrainer
from src import merge_metadata_and_features

class TestXGBTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 這段只會跑一次
        cls.task_name = 'play years'
        cls.trainer = XGBTrainer(task_name=cls.task_name, use_gpu=False)

        train_df_path = root_dir / "tests" / "train_data.csv"
        if train_df_path.exists():
            cls.train_df = pd.read_csv(train_df_path)
        else:
            meta_path = root_dir / "data" / "raw" / "train_info.csv"
            feature_folder = root_dir / "data" / "processed" / "train_features"
            cls.train_df = merge_metadata_and_features(meta_path, feature_folder)
            cls.train_df.to_csv(root_dir / "tests" / "train_data.csv", index=False)

        test_df_path = root_dir / "tests" / "test_data.csv"
        if test_df_path.exists():
            cls.test_df = pd.read_csv(test_df_path)
        else:
            meta_path = root_dir / "data" / "raw" / "test_info.csv"
            feature_folder = root_dir / "data" / "processed" / "test_features"
            cls.test_df = merge_metadata_and_features(meta_path, feature_folder)
            cls.test_df.to_csv(root_dir / "tests" / "test_data.csv", index=False)

    def test_xgb_trainer(self):
        task_name = 'play years'
        trainer = XGBTrainer(task_name=task_name, use_gpu=False)
        self.assertIsNotNone(trainer)

    def test_train_and_predict(self):
        self.trainer.auto_optimize(self.train_df)
        self.assertIsNotNone(self.trainer.models)
        
        self.trainer.set_test(self.test_df)
        model_name = self.trainer.show_models()[0]
        preds_df = self.trainer.predict(model_name=model_name)
        print(preds_df)
        self.assertIsInstance(preds_df, pd.DataFrame)
        self.assertGreater(len(preds_df), 0)
       
    @unittest.skip("not yet implemented")
    def test_load_model(self):
        # Assuming we have a model file to load
        # trainer.load_model("test_model", "path/to/model/file")
        # self.assertIn("test_model", trainer.models)
        pass

    @unittest.skip("not yet implemented")
    def test_save_model(self):
        # Assuming we have a model to save
        # trainer.save_model("test_model", "path/to/save/model")
        # self.assertTrue(os.path.exists("path/to/save/model"))
        pass
    def test_copy_model(self):
        # Assuming we have a model to copy
        # trainer.copy_model("test_model", "copied_model")
        # self.assertIn("copied_model", trainer.models)
        pass

    @unittest.skip("not yet implemented")
    def test_set_train(self):
        # Assuming we have a DataFrame for training
        # train_df = pd.DataFrame(...)
        # trainer.set_train(train_df)
        # self.assertEqual(trainer.train_data, train_df)
        pass

    @unittest.skip("not yet implemented")
    def test_set_val(self):
        # Assuming we have a DataFrame for validation
        # val_df = pd.DataFrame(...)
        # trainer.set_val(val_df)
        # self.assertEqual(trainer.val_data, val_df)
        pass

    @unittest.skip("not yet implemented")
    def test_set_test(self):
        # Assuming we have a DataFrame for testing
        # test_df = pd.DataFrame(...)
        # trainer.set_test(test_df)
        # self.assertEqual(trainer.test_data, test_df)
        pass

if __name__ == "__main__":
    unittest.main()