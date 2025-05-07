import matplotlib
matplotlib.use("Agg")  # 非 GUI 繪圖，節省記憶體

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_features(txt_path, feature_dir, image_dir, unique_id, cut_points, N=5):
    os.makedirs(os.path.join(image_dir, 'Trend'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'Seasonal'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'Resid'), exist_ok=True)

    data = np.loadtxt(txt_path)
    df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])

    # 加入合成向量
    df['a'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['g'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

    if len(cut_points) < 2 * N + 1:
        print(f"Skipping {unique_id} due to insufficient cut_points.")
        return

    start_idx = cut_points[N]
    end_idx = cut_points[-N]
    T = int(np.mean(np.diff(cut_points[N:-N])))

    feature_dict = {}
    components = {'trend': {}, 'seasonal': {}, 'resid': {}}

    for col in df.columns:
        signal = df[col][start_idx:end_idx].reset_index(drop=True)
        result = seasonal_decompose(signal, model='additive', period=T, extrapolate_trend='freq')

        for comp_name in ['trend', 'seasonal', 'resid']:
            comp_data = getattr(result, comp_name)
            components[comp_name][col] = comp_data

            # 統計特徵
            feature_dict[f'{col}_{comp_name}_mean'] = comp_data.mean()
            feature_dict[f'{col}_{comp_name}_std'] = comp_data.std()
            feature_dict[f'{col}_{comp_name}_var'] = comp_data.var()
            feature_dict[f'{col}_{comp_name}_max'] = comp_data.max()
            feature_dict[f'{col}_{comp_name}_min'] = comp_data.min()

    # 畫每個 component 的所有 signal
    for comp_name, series_dict in components.items():
        plt.figure(figsize=(12, 6))
        for sig_name, series in series_dict.items():
            plt.plot(series, label=sig_name)
        plt.title(f"{comp_name.capitalize()} - {unique_id}")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(image_dir, comp_name.capitalize(), f"{unique_id}.png")
        plt.savefig(save_path)
        plt.close()

    output_csv_path = os.path.join(feature_dir, f"{unique_id}.csv")
    pd.DataFrame([feature_dict]).to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # Example usage
    txt_path = r"/home/yanxunzhou/TBrain_SmartPing/data/raw/39_Training_Dataset/train_data/1.txt"
    output_csv_dir = r"/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/TimeFreqFeatures/Results3/features"
    output_img_dir = r"/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/TimeFreqFeatures/Results3/images"
    unique_id = "1"
    cut_points = [0, 61,  122,  183,  244,  305,  366,  428,  489,  550,  611,  672,  733,  794,  856,  917,  978, 1039, 1100, 1161, 1222, 1284, 1345, 1406, 1467, 1528, 1589, 1651]  # Example cut points
    
    generate_features(txt_path, output_csv_dir, output_img_dir, unique_id, cut_points)
    print(f"Features saved to {output_csv_dir} and image saved to {output_img_dir}")