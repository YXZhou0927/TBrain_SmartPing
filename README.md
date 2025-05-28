# 專案名稱

## 簡要說明 & 快速上手步驟

1. 從 TBrain 競賽網址下載 Train & Test Data。
2. 將資料全部解壓縮至 `data/raw` 資料夾下。
3. 執行 `scripts/process_data.py` 產生特徵資料至 `data/processed`。
4. 執行 `scripts/run_xgbtrainer.py` 一鍵訓練模型並產出 `submission_{時間戳記}.csv` 至 `outputs/submissions`。
5. 將 submission csv file 上傳至 TBrain 競賽網址查看成績。

## 更新紀錄

- **2025-05-29**
  - 新增 `scripts/process_data.py`：支援一鍵處理原始資料並產出特徵。
  - 新增 `scripts/run_xgbtrainer.py`：整合模型訓練與 submission 輸出流程。
  - 統一訓練結果輸出格式為 `submission_{時間戳記}.csv`。
  - 資料夾結構統一整理為：
    - `data/raw/`：放置原始資料（Train & Test）。
    - `data/processed/`：儲存處理後的特徵資料。
    - `outputs/submissions/`：儲存預測結果檔案。
