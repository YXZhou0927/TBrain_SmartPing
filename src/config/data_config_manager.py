from typing import List, Dict, TypedDict, Union
import json
import pandas as pd
from pathlib import Path


class TargetEncoding(TypedDict):
    classes: List[Union[int, str]]
    one_hot_columns: List[str]

class EncoderInfo(TypedDict):
    encoder_type: str
    handle_unknown: str

class FullConfig(TypedDict):
    one_hot_encoding: Dict[str, TargetEncoding]
    encoder_info: EncoderInfo
    scaler_type: Dict[str, str]

class DataConfigManager:
    def __init__(self):
        config_path = Path(__file__).resolve().parent / "data_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: FullConfig = json.load(f)

    def encode(self, target: str, value: Union[int, str]) -> List[int]:
        classes = self.config["one_hot_encoding"][target]["classes"]
        if value not in classes:
            if self.config["encoder_info"]["handle_unknown"] == "zero_pad":
                return [0] * len(classes)
            else:
                raise ValueError(f"Unknown value '{value}' for target '{target}'")
        return [1 if v == value else 0 for v in classes]

    def decode(self, target: str, one_hot: List[int]) -> Union[int, str, None]:
        classes = self.config["one_hot_encoding"][target]["classes"]
        if sum(one_hot) == 0:
            return None
        if one_hot.count(1) != 1:
            raise ValueError(f"Invalid one-hot vector: {one_hot}")
        return classes[one_hot.index(1)]

    def get_columns(self, target: str) -> List[str]:
        return self.config["one_hot_encoding"][target]["one_hot_columns"]
    
    def get_scaler_type(self, target: str) -> str:
        scaler_config = self.config.get("scaler_type", {})
        if target not in scaler_config:
            raise ValueError(f"Scaler type not defined for target '{target}'")
        return scaler_config[target]

    def encode_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame(index=df.index)
        encoded_any = False

        for target in self.config["one_hot_encoding"]:
            if target not in df.columns:
                continue
            encoded_cols = [self.encode(target, val) for val in df[target]]
            col_names = self.get_columns(target)
            encoded_df = pd.DataFrame(encoded_cols, columns=col_names, index=df.index)
            output_df = pd.concat([output_df, encoded_df], axis=1)
            encoded_any = True

        if not encoded_any:
            raise ValueError("No matching target columns found in DataFrame for encoding.")
        return output_df

    def decode_row(self, row_dict: Dict[str, Union[int, float]]) -> Dict[str, Union[int, str, None]]:
        result = {}
        for target in self.config["one_hot_encoding"]:
            one_hot_cols = self.get_columns(target)
            if not any(col in row_dict for col in one_hot_cols):
                continue
            one_hot_vector = [row_dict.get(col, 0) for col in one_hot_cols]
            result[target] = self.decode(target, one_hot_vector)
        return result

    def decode_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        decoded_rows = []
        for _, row in df.iterrows():
            decoded = self.decode_row(row.to_dict())
            if decoded:
                decoded_rows.append(decoded)

        if not decoded_rows:
            raise ValueError("No matching one-hot columns found in DataFrame for decoding.")
        return pd.DataFrame(decoded_rows, index=df.index[:len(decoded_rows)])

    def get_label_encoded_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame(index=df.index)
        any_encoded = False

        for target in self.config["one_hot_encoding"]:
            if target not in df.columns:
                continue
            classes = self.config["one_hot_encoding"][target]["classes"]
            class_to_index = {v: i for i, v in enumerate(classes)}
            encoded_series = df[target].map(class_to_index)

            if encoded_series.isnull().any():
                unknown_vals = df[target][encoded_series.isnull()].unique()
                raise ValueError(f"Unknown class values in column '{target}': {unknown_vals}")

            output_df[target] = encoded_series.astype(int)
            any_encoded = True

        if not any_encoded:
            raise ValueError("No matching target columns found in DataFrame for label encoding.")
        return output_df


if __name__ == "__main__":
    cfg = DataConfigManager()

    print("\n====== ✅ 測試 1: 全欄位編碼/解碼 ======")
    df_full = pd.DataFrame({
        "gender": [1, 2],
        "hold racket handed": [2, 1],
        "play years": [0, 2],
        "level": [5, 3]
    })
    print("原始資料：")
    print(df_full)

    onehot_df_full = cfg.encode_batch(df_full)
    print("One-hot 編碼結果：")
    print(onehot_df_full)

    decoded_df_full = cfg.decode_batch(onehot_df_full)
    print("解碼結果：")
    print(decoded_df_full)

    print("\n====== ✅ 測試 2: 僅部分欄位編碼（缺少 level） ======")
    df_partial = df_full[["gender", "play years"]]
    onehot_partial = cfg.encode_batch(df_partial)
    print(onehot_partial)

    print("\n====== ✅ 測試 3: 僅部分 one-hot 欄位解碼（缺少 gender） ======")
    partial_onehot = onehot_df_full.drop(columns=["gender_1", "gender_2"])
    decoded_partial = cfg.decode_batch(partial_onehot)
    print(decoded_partial)

    print("\n====== ❌ 測試 4: 完全沒 target 欄位（編碼） ======")
    try:
        df_empty = pd.DataFrame({"some_other_col": [1, 2]})
        cfg.encode_batch(df_empty)
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ❌ 測試 5: 完全沒 one-hot 欄位（解碼） ======")
    try:
        bad_onehot = pd.DataFrame({"some_col": [0, 1]})
        cfg.decode_batch(bad_onehot)
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ❌ 測試 6: 編碼遇到未知類別（level=999） ======")
    try:
        df_invalid = pd.DataFrame({
            "gender": [1],
            "hold racket handed": [1],
            "play years": [2],
            "level": [999]  # invalid class
        })
        cfg.encode_batch(df_invalid)
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ❌ 測試 7: 解碼時 one-hot 向量錯誤（多個 1） ======")
    try:
        broken_row = onehot_df_full.iloc[0].copy()
        broken_row["level_3"] = 1
        broken_row["level_5"] = 1  # now two 1s in level
        cfg.decode_row(broken_row.to_dict())
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ✅ 測試 8: 取得 label encoded labels for XGBoost ======")
    df_label = pd.DataFrame({
        "gender": [1, 2, 2],
        "level": [2, 5, 4]
    })
    label_encoded = cfg.get_label_encoded_targets(df_label)
    print(label_encoded)

    print("\n====== ❌ 測試 9: label encoding 遇到未知類別（level=999） ======")
    try:
        df_bad_label = pd.DataFrame({"level": [3, 999]})
        cfg.get_label_encoded_targets(df_bad_label)
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ❌ 測試 10: 沒有任何 target 欄位 ======")
    try:
        df_empty_label = pd.DataFrame({"abc": [1, 2]})
        cfg.get_label_encoded_targets(df_empty_label)
    except ValueError as e:
        print("捕捉到錯誤：", e)

    print("\n====== ✅ 測試 11: get_scaler_type 功能測試 ======")
    try:
        print("gender 的 scaler_type:", cfg.get_scaler_type("gender"))  # 預期 minmax
        print("level 的 scaler_type:", cfg.get_scaler_type("level"))    # 預期 minmax
        print("未知欄位的 scaler_type:", cfg.get_scaler_type("unknown"))  # 應報錯
    except ValueError as e:
        print("捕捉到錯誤：", e)