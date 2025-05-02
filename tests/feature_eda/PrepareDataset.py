import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import threading

# Global paths
meta_file_path = ""
data_folder_path = ""

# Main window
root = tk.Tk()
root.title("Training Data Generator")

# Display variables
meta_var = tk.StringVar(value="Not selected")
data_var = tk.StringVar(value="Not selected")

def select_meta_file():
    global meta_file_path
    meta_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if meta_file_path:
        meta_var.set(f"{os.path.basename(meta_file_path)} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

def select_data_folder():
    global data_folder_path
    data_folder_path = filedialog.askdirectory()
    if data_folder_path:
        data_var.set(f"{os.path.basename(data_folder_path)} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

def generate_training_data():
    if not meta_file_path or not data_folder_path:
        messagebox.showerror("Error", "Please select meta CSV and data folder first.")
        return

    output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not output_path:
        return

    try:
        df = pd.read_csv(meta_file_path)
        df = df[['unique_id', 'player_id', 'mode', 'gender', 'hold racket handed', 'play years', 'level']]
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]  # 替換欄位名稱中的空白
        feature_rows = []
        all_column_names = []

        progress_label.config(text="Generating features...")
        root.update()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ncols=80):
            uid = row['unique_id']
            file_path = os.path.join(data_folder_path, f"{uid}.csv")
            if os.path.exists(file_path):
                try:
                    matrix_df = pd.read_csv(file_path)  # DataFrame with header
                    columns = matrix_df.columns
                    matrix = matrix_df.values

                    flattened = []
                    colnames = []
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            flattened.append(matrix[i, j])
                            colnames.append(f"feat_{i}_{j}_{columns[j]}")

                    feature_rows.append(flattened)

                    if len(colnames) > len(all_column_names):
                        all_column_names = colnames  # 更新最大長度與欄位名稱

                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    feature_rows.append([np.nan])
            else:
                feature_rows.append([np.nan])

        # 對齊長度（補 NaN）
        max_len = len(all_column_names)
        for i in range(len(feature_rows)):
            row = feature_rows[i]
            if len(row) < max_len:
                feature_rows[i] = row + [np.nan] * (max_len - len(row))

        feature_df = pd.DataFrame(feature_rows, columns=all_column_names)
        output_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        output_df.to_csv(output_path, index=False)

        progress_label.config(text="Completed!")
        messagebox.showinfo("Done", "Training data has been generated.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def start_generation_thread():
    threading.Thread(target=generate_training_data).start()

# GUI Layout
tk.Button(root, text="Select Meta CSV", command=select_meta_file).grid(row=0, column=0, padx=5, pady=5, sticky="w")
tk.Label(root, textvariable=meta_var, anchor="w", width=60).grid(row=0, column=1, padx=5, pady=5, sticky="w")

tk.Button(root, text="Select Data Folder", command=select_data_folder).grid(row=1, column=0, padx=5, pady=5, sticky="w")
tk.Label(root, textvariable=data_var, anchor="w", width=60).grid(row=1, column=1, padx=5, pady=5, sticky="w")

tk.Button(root, text="Generate Training Data", command=start_generation_thread, bg="lightgreen")\
    .grid(row=2, column=0, columnspan=2, padx=5, pady=20)

progress_label = tk.Label(root, text="", fg="blue")
progress_label.grid(row=3, column=0, columnspan=2)

root.mainloop()
