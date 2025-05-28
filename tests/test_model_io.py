import os
import tempfile
import unittest
import xgboost as xgb
import numpy as np
import pickle
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent  # Assuming the script is in tests directory
sys.path.append(str(root_dir))

from src import XGBTrainer
from src.models.model_registry import save_model_file, load_model_file


class TestModelIO(unittest.TestCase):
    def setUp(self):
        # Create temp directory
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_filename = os.path.join(self.tmpdir.name, "test_model")

        # Prepare dummy training data
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        dtrain = xgb.DMatrix(self.X, label=self.y)

        # Train simple model
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        }
        self.model = xgb.train(param, dtrain, num_boost_round=3)

        # Dict to save
        self.model_data = {
            "class_name": "XGBTrainer",
            "model": self.model,
            "eval_method": "logloss",
            "score": 0.1234
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_save_and_load_model_file(self):
        # Save the model dict
        save_model_file(self.model_data, self.base_filename)

        # Load it back
        loaded = load_model_file(self.base_filename)

        # Check content
        self.assertIsInstance(loaded, dict)
        self.assertIn("model", loaded)
        self.assertIsInstance(loaded["model"], xgb.Booster)
        self.assertEqual(loaded["class_name"], "XGBTrainer")
        self.assertEqual(loaded["eval_method"], "logloss")
        self.assertAlmostEqual(loaded["score"], 0.1234, places=6)

        # Check model is usable
        dpred = xgb.DMatrix(self.X)
        preds = loaded["model"].predict(dpred)
        self.assertEqual(preds.shape[0], 4)
        print("Predictions:", preds)


if __name__ == "__main__":
    unittest.main()