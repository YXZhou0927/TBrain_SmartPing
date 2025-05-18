import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import logging

logging.basicConfig(filename="add_poly_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class FeatureAdderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Add Polynomial Features")

        self.folder_path = ""
        self.top_feature_path = ""
        self.top_features_df = None

        # GUI layout
        tk.Button(root, text="Select Folder with {unique_id}.csv", command=self.select_folder).pack(pady=5)
        tk.Button(root, text="Select Top Feature Combination CSV", command=self.select_top_csv).pack(pady=5)
        tk.Button(root, text="Add New Features", command=self.add_features).pack(pady=10)

        self.log_text = tk.Text(root, height=15, width=70)
        self.log_text.pack()

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.log(f"Selected folder: {self.folder_path}")

    def select_top_csv(self):
        self.top_feature_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.top_feature_path:
            try:
                self.top_features_df = pd.read_csv(self.top_feature_path)
                required_cols = {"new_feature", "feature_1", "feature_2"}
                if not required_cols.issubset(self.top_features_df.columns):
                    raise ValueError("CSV must contain columns: new_feature, feature_1, feature_2")
                self.log(f"Loaded top combinations from: {self.top_feature_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read top feature CSV: {e}")

    def add_features(self):
        if not self.folder_path or self.top_features_df is None:
            messagebox.showerror("Error", "Please select both the folder and the top features CSV.")
            return

        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]

        for file in csv_files:
            file_path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(file_path)
                for _, row in self.top_features_df.iterrows():
                    new_col = row["new_feature"]
                    f1 = row["feature_1"]
                    f2 = row["feature_2"]

                    if f1 in df.columns and f2 in df.columns:
                        df[new_col] = df[f1] * df[f2]
                        self.log(f"[{file}] Added feature: {new_col} = {f1} * {f2}")
                    else:
                        self.log(f"[{file}] Missing features: {f1}, {f2}")

                df.to_csv(file_path, index=False)
                self.log(f"[{file}] Saved with new features.")
            except Exception as e:
                self.log(f"Failed to process {file}: {e}")

        messagebox.showinfo("Done", "All CSV files updated successfully.")

    def log(self, msg):
        logging.info(msg)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureAdderApp(root)
    root.mainloop()
