import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from tests.feature_eda.TimeSeriesFeatures.Code.generate_features import generate_features
import numpy as np
import re
import logging

# setup logging and record the log
log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "feature_generation.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("Feature generation started.")
logging.info("Log file created at: %s", log_file)

def parse_cut_point(cut_str):
    # 移除中括號與多餘空白，使用 re 將數字提取出來
    return list(map(int, re.findall(r'\d+', str(cut_str))))

# === 全局變數 ===
meta_path = r"/home/yanxunzhou/TBrain_SmartPing/data/raw/39_Training_Dataset/train_info.csv"
data_folder = r"/home/yanxunzhou/TBrain_SmartPing/data/raw/39_Training_Dataset/train_data"
output_folder = r"/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/TimeSeriesFeatures/Results5"

root = tk.Tk()
root.title("Wavelet Feature Generator")

meta_var = tk.StringVar(value="Not selected")
data_var = tk.StringVar(value="Not selected")
output_var = tk.StringVar(value="Not selected")

# === GUI 控制 ===
def select_meta():
    global meta_path
    meta_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if meta_path:
        meta_var.set(os.path.basename(meta_path))

def select_data_folder():
    global data_folder
    data_folder = filedialog.askdirectory()
    if data_folder:
        data_var.set(os.path.basename(data_folder))

def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory()
    if output_folder:
        output_var.set(os.path.basename(output_folder))

# === 執行核心邏輯 ===
def run_generation_parallel():
    if not (meta_path and data_folder and output_folder):
        messagebox.showerror("Error", "Please select all required paths.")
        return

    feature_dir = os.path.join(output_folder, "features")
    image_dir = os.path.join(output_folder, "images")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_csv(meta_path)
    df['unique_id'] = df['unique_id'].astype(str)
    df = df.drop(columns=['cut_point'], errors='ignore')  # 移除不必要的欄位
    #df = df[['unique_id', 'player_id', 'mode', 'gender', 'hold_racket_handed', 'play_years', 'level', 'cut_point']]

    futures = []
    max_workers = os.cpu_count() or 4

    progress_bar['maximum'] = len(df)
    progress_bar['value'] = 0
    progress_label.config(text="Starting...")

    def task():
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for _, row in df.iterrows():
                uid = row['unique_id']
                txt_file = os.path.join(data_folder, f"{uid}.txt")
                #out_csv = os.path.join(feature_dir, f"{uid}.csv")
                #out_png = os.path.join(image_dir, f"{uid}.png")

                logging.info(f"Processing {uid}...")
                if os.path.exists(txt_file):
                    logging.info(f"Input file: {txt_file}")
                    logging.info(f"Output feature dir: {feature_dir}")
                    logging.info(f"Output image dir: {image_dir}")
                
                    #cut_point = parse_cut_point(row['cut_point'])  # 轉為 list[int]
                    futures.append(executor.submit(
                        generate_features,
                        uid, txt_file, feature_dir, None # None -> Do not save images
                    ))
                else:
                    logging.warning(f"File not found: {txt_file}")
                    messagebox.showwarning("Warning", f"File not found: {txt_file}")   

            for i, _ in enumerate(as_completed(futures)):
                progress_bar['value'] = i + 1
                progress_label.config(text=f"Processing: {i+1}/{len(futures)}")
                root.update_idletasks()

        progress_label.config(text="Completed!")
        messagebox.showinfo("Done", "Wavelet features and images generated!")

    threading.Thread(target=task).start()

# === GUI Layout ===
tk.Button(root, text="Select Meta CSV", command=select_meta).grid(row=0, column=0, sticky="w", padx=5, pady=5)
tk.Label(root, textvariable=meta_var).grid(row=0, column=1, sticky="w")

tk.Button(root, text="Select Data Folder", command=select_data_folder).grid(row=1, column=0, sticky="w", padx=5, pady=5)
tk.Label(root, textvariable=data_var).grid(row=1, column=1, sticky="w")

tk.Button(root, text="Select Output Folder", command=select_output_folder).grid(row=2, column=0, sticky="w", padx=5, pady=5)
tk.Label(root, textvariable=output_var).grid(row=2, column=1, sticky="w")

tk.Button(root, text="Generate Features", command=run_generation_parallel, bg="lightgreen")\
    .grid(row=3, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(root, length=300, mode='determinate')
progress_bar.grid(row=4, column=0, columnspan=2, pady=5)

progress_label = tk.Label(root, text="", fg="blue")
progress_label.grid(row=5, column=0, columnspan=2)

root.mainloop()
