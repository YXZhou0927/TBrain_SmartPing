import pickle
import xgboost as xgb
from pathlib import Path

def save_model_file(model_data: dict, filename: str):
    """
    Save model dictionary. Booster is saved separately to avoid pickle error.
    """
    filename = Path(filename)
    base = filename.with_suffix('')  # 去掉副檔名

    # 1. 儲存模型本體
    model_file = base.with_suffix(".model")
    model_data["model"].save_model(str(model_file))

    # 2. 拷貝 dict 並去掉 model 欄位
    data_copy = model_data.copy()
    data_copy["model_file"] = model_file.name
    data_copy.pop("model")

    # 3. 儲存 metadata
    meta_file = base.with_suffix(".pkl")
    with open(meta_file, "wb") as f:
        pickle.dump(data_copy, f)

def load_model_file(filename: str) -> dict:
    """
    Load model dictionary. Will load Booster from separate file.
    """
    filename = Path(filename)
    base = filename.with_suffix('')

    # 1. 載入 metadata
    meta_file = base.with_suffix(".pkl")
    with open(meta_file, "rb") as f:
        model_data = pickle.load(f)

    # 2. 載入 Booster 模型
    model_file = base.parent / model_data["model_file"]
    booster = xgb.Booster()
    booster.load_model(str(model_file))
    model_data["model"] = booster

    return model_data
