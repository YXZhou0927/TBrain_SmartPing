import os
import shutil

tmp = '10'
# 設定來源與目標資料夾
source_folder = "/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/result2"
destination_folder = f"/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/mode{tmp}"

# 確保目標資料夾存在
os.makedirs(destination_folder, exist_ok=True)

# 遍歷來源資料夾
for file_name in os.listdir(source_folder):
    if f"_{tmp}.png" in file_name.lower():  # 檢查是否包含 "1.png"
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        shutil.copy2(source_path, destination_path)  # 複製檔案並保留原始時間戳
        print(f"已複製: {file_name}")

print("所有符合條件的檔案已複製完成！")
