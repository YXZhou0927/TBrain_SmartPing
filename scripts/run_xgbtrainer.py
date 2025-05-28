from pathlib import Path
import pandas as pd
import sys
import time

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent  # Assuming the script is in tests directory
sys.path.append(str(ROOT_DIR))
from src import XGBTrainer
from src import merge_metadata_and_features

trainers = {
    "gender": XGBTrainer('gender'),
    "hold racket handed": XGBTrainer('hold racket handed'),
    "play years": XGBTrainer('play years'),
    "level": XGBTrainer('level')
}
outputs = {}

info_dir = ROOT_DIR / "data" / "raw"
feature_dir = ROOT_DIR / "data" / "processed"
cache_dir = ROOT_DIR / "scripts" / "cache"
output_dir = ROOT_DIR / "outputs" / "submissions"

train_path = cache_dir / "train_data.csv"
test_path = cache_dir / "test_data.csv"
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"submission_{timestamp}.csv"

cache_dir.mkdir(parents=True, exist_ok=True)
if train_path.exists():
    train_df = pd.read_csv(train_path)
else:
    train_df = merge_metadata_and_features(info_dir / "train_info.csv", feature_dir / "train_features")
    train_df.to_csv(train_path)
if test_path.exists():
    test_df = pd.read_csv(test_path)
else:
    test_df = merge_metadata_and_features(info_dir / "test_info.csv", feature_dir / "test_features")
    test_df.to_csv(test_path)

final_score_list = []
for task_name, trainer in trainers.items():
    trainer.auto_optimize(train_df)
    model_names = trainer.show_models()
    preds_list = []
    score_list = []

    for model_name in model_names:
        preds = trainer.predict(model_name, test_df)  # 預期返回 DataFrame，欄位為 one-hot columns
        val_score = trainer.models[model_name]['score']
        preds_list.append(preds)
        if val_score is not None:
            score_list.append(val_score)

    if len(score_list) > 0:
        final_score_list.append(sum(score_list) / len(score_list))
    else:
        breakpoint()

    # 將多個模型預測結果取平均
    avg_preds = sum(preds_list) / len(preds_list)
    avg_preds.columns = preds_list[0].columns  # 保留欄位名稱
    outputs[task_name] = avg_preds

print(f"  Final Score: {sum(final_score_list) / len(final_score_list)}")

# 取得 unique_id 欄位（假設 test_df 中有）
submission_df = pd.DataFrame({'unique_id': test_df['unique_id']})

# 將所有任務的預測結果 DataFrame 水平合併
for task_name, preds in outputs.items():
    submission_df = pd.concat([submission_df, preds], axis=1)

# 格式化 submission
submission_df = submission_df.drop(columns=['gender_2', 'hold racket handed_2'], errors='ignore')
submission_df = submission_df.rename(columns={
    'gender_1': 'gender',
    'hold racket handed_1': 'hold racket handed'
})

# 儲存最終上傳檔案（你可以在這裡補上儲存路徑）
submission_df.to_csv(str(output_path), index=False)
