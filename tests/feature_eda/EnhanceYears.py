import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import logging
from functools import wraps
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif

# Logging setup
logging.basicConfig(filename="feature_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Logging decorator
def log_action(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Function '{func.__name__}' called with args={args[1:]}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

# Feature transformation
class FeatureTransformer:
    def __init__(self, df, target_column):
        self.df = df
        self.target = df[target_column]
        self.feature_cols = [col for col in df.columns if col.startswith("feat_")]
        self.X = df[self.feature_cols].copy()
    
    @log_action
    def apply_pca(self, n_components=5):
        pca = PCA(n_components=n_components)
        X_new = pca.fit_transform(self.X)
        return self._evaluate_mi(X_new, "PCA")

    @log_action
    def apply_ica(self, n_components=5):
        ica = FastICA(n_components=n_components, random_state=0)
        X_new = ica.fit_transform(self.X)
        return self._evaluate_mi(X_new, "ICA")

    @log_action
    def apply_poly(self, degree=2):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_new = poly.fit_transform(self.X)
        feature_names = poly.get_feature_names_out(self.feature_cols)
        return self._evaluate_mi(X_new, "PolynomialFeatures", feature_names)


    def _evaluate_mi(self, X_new, method_name, feature_names=None):
        mi = mutual_info_classif(X_new, self.target, discrete_features='auto', random_state=0)
        mi_series = pd.Series(mi, index=feature_names if feature_names is not None else range(len(mi)))
        top_scores = mi_series.sort_values(ascending=False).head(10)
        logging.info(f"{method_name} Top 10 MI scores:\n{top_scores}")
        return top_scores



# GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Engineering Tool")
        self.df = None

        tk.Button(root, text="Load CSV", command=self.load_csv).pack(pady=5)

        self.method_var = tk.StringVar(value="PCA")
        tk.Label(root, text="Select Method:").pack()
        for method in ["PCA", "ICA", "Poly"]:
            tk.Radiobutton(root, text=method, variable=self.method_var, value=method).pack(anchor='w')

        tk.Label(root, text="Target Column Name:").pack()
        self.target_entry = tk.Entry(root)
        self.target_entry.insert(0, "Play year")
        self.target_entry.pack(pady=5)

        tk.Button(root, text="Run Feature Analysis", command=self.run).pack(pady=10)

        self.result_text = tk.Text(root, height=15, width=60)
        self.result_text.pack()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.df = pd.read_csv(path)
        messagebox.showinfo("Success", f"CSV loaded:\n{path}")

    def run(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        target = self.target_entry.get()
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found in data.")
            return

        transformer = FeatureTransformer(self.df, target)

        method = self.method_var.get()
        if method == "PCA":
            mi_scores = transformer.apply_pca()
        elif method == "ICA":
            mi_scores = transformer.apply_ica()
        elif method == "Poly":
            mi_scores = transformer.apply_poly()
        else:
            messagebox.showerror("Error", "Unknown method selected.")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"{method} Top 10 MI scores:\n\n")
        for name, score in mi_scores.items():
            self.result_text.insert(tk.END, f"{name}: {score:.4f}\n")

        messagebox.showinfo("Completed", f"{method} feature analysis completed!")


# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
