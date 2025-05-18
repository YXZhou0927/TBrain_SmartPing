import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import psutil
import gc
import matplotlib
matplotlib.use("Agg") # Use non-GUI backend for matplotlib

# ============================================
# GUI Application: Plot time series of training data
# Assumes a folder containing train_info.csv and train_data subfolder with {unique_id}.txt
# ============================================

def select_input_folder():
    """
    Let user select the main folder containing train_info.csv and train_data subfolder.
    """
    global input_folder
    input_folder = filedialog.askdirectory(title="Select Training Data Folder")
    if input_folder:
        info_label.config(text=f"Selected folder: {input_folder}")


def plot_time_series():
    """
    Read train_info.csv and plot 6-axis time series for each unique_id txt file.
    Save figures to the user-selected output folder.
    """
    if not input_folder:
        messagebox.showerror("Error", "Please select the training data folder first!")
        return
   
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        return

    try:
        y_min = float(ymin_entry.get())
        y_max = float(ymax_entry.get())
        x_max = float(xmax_entry.get())
        if y_min >= y_max:
            raise ValueError("Y-axis min must be less than max.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric limits!")
        return

    info_csv = os.path.join(input_folder, "train_info.csv")
    data_folder = os.path.join(input_folder, "train_data")
    try:
        df_info = pd.read_csv(info_csv)
        df_info.columns = [col.strip().replace(' ', '_') for col in df_info.columns]  # 替換欄位名稱中的空白
    except Exception as e:
        messagebox.showerror("Error", f"Unable to read train_info.csv:\n{e}")
        return
    
    process = psutil.Process(os.getpid())

    for row in tqdm(df_info.itertuples(index=False), total=len(df_info), desc="Processing files"):
        uid = row.unique_id
        player_id = row.player_id
        mode = row.mode
        gender = row.gender
        handed = row.hold_racket_handed
        years = row.play_years
        level = row.level

        txt_path = os.path.join(data_folder, f"{uid}.txt")
        if not os.path.isfile(txt_path):
            print(f"Warning: {uid}.txt not found, skipped.")
            continue

        data = None
        data = pd.read_csv(txt_path, sep='\s+', header=None,
                        names=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
        data = data[['Ax', 'Ay', 'Az']]

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in data.columns:
            ax.plot(data[col], label=col)

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, x_max)
        ax.set_title(
            f"ID: {uid}  Player: {player_id}  Mode: {mode}\n"
            f"Gender: {gender}  Handed: {handed}  Years: {years}  Level: {level}"
        )
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Signal Value')
        ax.legend(loc='upper right')

        save_name = f"{uid}_{player_id}_{mode}.png"
        save_path = os.path.join(output_folder, save_name)
        fig.savefig(save_path)

        fig.clf()
        plt.clf()
        plt.cla()
        plt.close(fig)
        del data, fig, ax
        gc.collect()

        print(f"Memory usage after {uid}: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    messagebox.showinfo("Done", "All time series plots have been generated and saved!")


if __name__ == '__main__':
    input_folder = ''

    root = tk.Tk()
    root.title('Training Data Time Series Plotter')
    root.geometry('450x200')

    btn1 = tk.Button(root, text='Select Training Data Folder', command=select_input_folder)
    btn1.pack(pady=10)

    frame = tk.Frame(root)
    tk.Label(frame, text='Y-axis Min:').grid(row=0, column=0)
    ymin_entry = tk.Entry(frame, width=10)
    ymin_entry.grid(row=0, column=1, padx=5)
    tk.Label(frame, text='Y-axis Max:').grid(row=0, column=2)
    ymax_entry = tk.Entry(frame, width=10)
    ymax_entry.grid(row=0, column=3, padx=5)
    tk.Label(frame, text='X-axis: Max:').grid(row=0, column=4)
    xmax_entry = tk.Entry(frame, width=10)
    xmax_entry.grid(row=0, column=5, padx=5)
    frame.pack(pady=5)
    
    btn2 = tk.Button(root, text='Plot Time Series', command=plot_time_series)
    btn2.pack(pady=10)

    info_label = tk.Label(root, text='No folder selected')
    info_label.pack(pady=5)

    root.mainloop()
