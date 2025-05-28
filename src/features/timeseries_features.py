import matplotlib
matplotlib.use("Agg")  # 非 GUI 繪圖，節省記憶體

from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
from scipy.signal import find_peaks
from typing import List, Tuple
import logging

def perform_fft(unique_id, signal, sampling_rate=85.0, image_dir=None):
    """
    Perform FFT on the signal and plot the spectrum.
    :param unique_id: Unique identifier for the input data
    :param signal: Input signal
    :param sampling_rate: Sampling rate of the signal
    :param image_dir: Directory to save the FFT plot
    :return: Frequency, Magnitude, FFT result
    """
    # FFT
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)
    magnitude = np.abs(fft_result)
    half = len(freq) // 2

    # plotting
    if image_dir is not None:
        os.makedirs(os.path.join(image_dir, 'FFT'), exist_ok=True)
        fft_img_path = os.path.join(image_dir, 'FFT', f"{unique_id}.png")
        plt.figure(figsize=(14, 8))
        plt.plot(freq[:half], magnitude[:half])
        plt.title("FFT Spectrum\n"
                "Frequency: {:.3f} Hz | T={:.2f}s | ~{:.1f} repeats".format(
                    freq[np.argmax(magnitude[1:]) + 1],  # skip DC
                    1.0 / freq[np.argmax(magnitude[1:]) + 1] if freq[np.argmax(magnitude[1:]) + 1] != 0 else np.inf,
                    len(signal) / (1.0 / freq[np.argmax(magnitude[1:]) + 1]) / sampling_rate if freq[np.argmax(magnitude[1:]) + 1] != 0 else 0
                ) if len(signal) > 0 else "No data"  
                )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.savefig(fft_img_path)
        plt.close()

    # return only the positive frequencies
    return freq[:half], magnitude[:half], fft_result

def bandpass_ifft_feature(fft_result, freq, center_idx, lower_ratio=0.25, upper_ratio=8.0):
    mask = np.zeros_like(fft_result, dtype=complex)
    lower_bound = int(center_idx * lower_ratio)
    upper_bound = int(center_idx * upper_ratio)
    mask[lower_bound:upper_bound + 1] = fft_result[lower_bound:upper_bound + 1]
    if lower_bound > 0:
        mask[-upper_bound:-lower_bound + 1] = fft_result[-upper_bound:-lower_bound + 1]
    feature_signal = np.real(np.fft.ifft(mask))
    return feature_signal

def get_cut_points(unique_id: str, signal: np.ndarray, top_threshold: float = 0.2, image_dir: str = None) -> Tuple[List[int], int]:
    """
    Perform cut points detection on the input signal.
    :param signal: Input signal
    :param top_threshold: 峰值檢測的閾值
    :param image_dir: 圖片儲存的目錄
    :return: cut points 的 index list, 週期 T
    """
    signal_smooth = np.convolve(signal, np.ones(50)/50, mode='same')
    signal_demeaned = signal_smooth - np.mean(signal_smooth)
    sampling_rate = 85.0
    freq, mag, fft = perform_fft(unique_id, signal_demeaned, sampling_rate=sampling_rate, image_dir=image_dir)
    
    # Dominant frequency analysis
    lower_idx = np.where(1.0 / freq[1:] > 1.2)[0][-1] + 1
    upper_idx = np.where(1.0 / freq[1:] < 0.7)[0][0] + 1
    idx = np.argmax(mag[lower_idx:upper_idx] * freq[lower_idx:upper_idx]**2) + lower_idx
    #idx = np.argmax(mag[1:]) + 1
    dom_freq = freq[idx]
    #dom_freq = 50 / 60
    dom_period = 1.0 / dom_freq if dom_freq != 0 else np.inf
    
    # 進行頻帶通濾波
    feature = bandpass_ifft_feature(fft, freq, idx, lower_ratio=0.8, upper_ratio=1.2)
    feature = feature / np.max(np.abs(feature))
    cut_points, _ = find_peaks(feature, height=top_threshold)

    if len(cut_points) > 2:
        # 計算 IQR
        value = signal_smooth[cut_points]
        q3 = np.percentile(value, 75)
        q1 = np.percentile(value, 25)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Step 1: 取出 valid_cut_points
        valid_cut_points = [cut for cut in cut_points if lower_bound < signal_smooth[cut] < upper_bound]

        # Step 2: 在 cut_points 中找出 valid_cut_points 連續出現三次的位置
        valid_set = set(valid_cut_points)
        consecutive_indices = []

        for i in range(len(cut_points) - 2):
            if (cut_points[i] in valid_set and
                cut_points[i+1] in valid_set and
                cut_points[i+2] in valid_set):
                consecutive_indices.append(i)

        # Step 3: 決定 min_cut_point 和 max_cut_point
        if consecutive_indices:
            min_cut_point = cut_points[consecutive_indices[0]]
            max_cut_point = cut_points[consecutive_indices[-1] + 2]
        else:
            min_cut_point = 0
            max_cut_point = len(signal) - 1

        # Step 4: 篩選 cut_points 落在 min 和 max 範圍內
        cut_points = cut_points[(cut_points >= min_cut_point) & (cut_points <= max_cut_point)]

    # Step 5: 如果 cut_points 少於 2 個，則使用 signal 的起始和結束位置
    if len(cut_points) < 2:
        cut_points = [0, len(signal) - 1]
    else:
        # 確保 cut_points 是升序的
        cut_points = sorted(cut_points)
        cut_points = [int(cut) for cut in cut_points]

    # Plotting
    if image_dir is not None:
        os.makedirs(os.path.join(image_dir, 'CutPoints'), exist_ok=True)
        cut_points_img_path = os.path.join(image_dir, 'CutPoints', f"{unique_id}.png")
        plt.figure(figsize=(14, 8))
        plt.plot(signal_demeaned / np.max(np.abs(signal)), label="|Signal| (demeaned)")
        plt.plot(feature / np.max(np.abs(feature)), label="|Feature| (demeaned)", linestyle="--")
        if cut_points is not None:
            plt.scatter(cut_points, feature[cut_points] / np.max(np.abs(feature)), color='red', label="Cut Points")
        duration = len(signal) / sampling_rate
        repeats_c = duration / dom_period if dom_period != np.inf else 0
        plt.title(
            "Cut Points Detection\n"
            f"Combined: {dom_freq:.3f} Hz | T={dom_period:.2f}s | ~{repeats_c:.1f} repeats"
            )
        plt.xlabel("Sample Index")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.savefig(cut_points_img_path)
        plt.close()

    return cut_points, int(dom_period * sampling_rate)

def generate_features(unique_id: str, txt_path: str, feature_dir: str, image_dir: str = None):
    """
    Generate timeseries features from the input signal data.
    :param txt_path: Path to the input signal data file
    :param feature_dir: Directory to save the generated features
    :param image_dir: Directory to save the generated images
    :param unique_id: Unique identifier for the input data
    """
    
    try:
        os.makedirs(feature_dir, exist_ok=True)
        if image_dir is not None:
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(os.path.join(image_dir, 'Signal'), exist_ok=True)
            os.makedirs(os.path.join(image_dir, 'Trend'), exist_ok=True)
            os.makedirs(os.path.join(image_dir, 'Seasonal'), exist_ok=True)
            os.makedirs(os.path.join(image_dir, 'Resid'), exist_ok=True)
            os.makedirs(os.path.join(image_dir, 'acf'), exist_ok=True)

        data = np.loadtxt(txt_path)
        df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        
        # Add 3-D radius and angle features
        df['a_radius'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['a_phi1'] = np.arctan2(df['ay'], df['ax'])
        df['a_theta1'] = np.arctan2(df['az'], np.sqrt(df['ax']**2 + df['ay']**2))
        # df['a_phi2'] = np.arctan2(df['az'], df['ax'])
        # df['a_theta2'] = np.arctan2(df['ay'], np.sqrt(df['ax']**2 + df['az']**2))
        # df['a_phi3'] = np.arctan2(df['az'], df['ay'])
        # df['a_theta3'] = np.arctan2(df['ax'], np.sqrt(df['ay']**2 + df['az']**2))
        df['g_radius'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
        df['g_phi1'] = np.arctan2(df['gy'], df['gx'])
        df['g_theta1'] = np.arctan2(df['gz'], np.sqrt(df['gx']**2 + df['gy']**2))
        # df['g_phi2'] = np.arctan2(df['gz'], df['gx'])
        # df['g_theta2'] = np.arctan2(df['gy'], np.sqrt(df['gx']**2 + df['gz']**2))
        # df['g_phi3'] = np.arctan2(df['gz'], df['gy'])
        # df['g_theta3'] = np.arctan2(df['gx'], np.sqrt(df['gy']**2 + df['gz']**2))


        a_energy = np.sum(df['a_radius']**2) / len(df['a_radius'])
        g_energy = np.sum(df['g_radius']**2) / len(df['g_radius'])
        df['combined'] = np.sqrt(df['a_radius']**2 / a_energy + df['g_radius']**2 / g_energy) * (df['a_radius'].max() + df['g_radius'].max()) / 2
        

        cut_points, T = get_cut_points(unique_id, df['combined'].values, top_threshold=0.2, image_dir=image_dir)
        
        start_idx = cut_points[0]
        end_idx = cut_points[-1]

        feature_dict = {}
        feature_dict['period'] = T
        components = {'trend': {}, 'seasonal': {}, 'resid': {}}

        #T = int(T * 1.5)
        for col in df.columns:
            signal = df[col][start_idx:end_idx].reset_index(drop=True)
            stl = STL(signal, period=T)
            # mstl = MSTL(signal, periods=(T, T+1))
            result = stl.fit()
            # result_df = pd.DataFrame({
            #     'trend': result.trend,
            #     'seasonal': result.seasonal[f'seasonal_{T}'],
            #     'resid': result.resid
            # })

            for comp_name in ['trend', 'seasonal', 'resid']:
                comp_data = getattr(result, comp_name)
                components[comp_name][col] = comp_data

                feature_dict[f'{col}_{comp_name}_mean'] = comp_data.mean()
                feature_dict[f'{col}_{comp_name}_std'] = comp_data.std()
                feature_dict[f'{col}_{comp_name}_var'] = comp_data.var()
                feature_dict[f'{col}_{comp_name}_max'] = comp_data.max()
                feature_dict[f'{col}_{comp_name}_min'] = comp_data.min()

                if comp_name == 'trend':
                    slope = np.polyfit(range(len(comp_data)), comp_data, 1)[0]
                    feature_dict[f'{col}_trend_slope'] = slope
                    feature_dict[f'{col}_trend_range'] = comp_data.max() - comp_data.min()
                    feature_dict[f'{col}_trend_energy'] = np.mean(comp_data**2)
                elif comp_name == 'seasonal':
                    ptp = comp_data.max() - comp_data.min()
                    feature_dict[f'{col}_seasonal_ptp'] = ptp
                    power = np.abs(np.fft.fft(comp_data))**2
                    power /= np.sum(power)
                    lag = T
                    feature_dict[f'{col}_seasonal_freq_entropy'] = entropy(power)
                    autocorr = comp_data.autocorr(lag=lag)
                    feature_dict[f'{col}_seasonal_autocorr1'] = autocorr if not pd.isna(autocorr) else 1.0
                elif comp_name == 'resid':
                    #rms = np.sqrt(np.mean(comp_data**2)) # It's equivalent to std
                    zcr = np.sum(np.diff(np.sign(comp_data)) != 0) / len(comp_data)
                    dx = np.diff(comp_data)
                    ddx = np.diff(dx)
                    complexity = np.std(ddx) / (np.std(dx) + 1e-6)
                    #feature_dict[f'{col}_resid_rms'] = rms # It's equivalent to std
                    feature_dict[f'{col}_resid_zero_crossing_rate'] = zcr
                    feature_dict[f'{col}_resid_complexity'] = complexity

        # === Plotting ===
        if image_dir is not None:
            plt.figure(figsize=(12, 6))
            for col in df.columns:
                plt.plot(df[col], label=col)
            plt.title(f"Raw Signals - {unique_id}")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(image_dir, 'Signal', f"{unique_id}.png")
            plt.savefig(save_path)
            plt.close()

            # Plot each component
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

            # === ACF + Ljung-Box ===
            #resid_all = pd.DataFrame(components['resid'])
            #resid_avg = resid_all.mean(axis=1).dropna()
            resid_avg = components['resid']['combined']
            lags = min(T, len(resid_avg) // 2)
            lb_result = acorr_ljungbox(resid_avg, lags=lags, return_df=True)
            p_value = lb_result["lb_pvalue"].iloc[-1]

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_acf(resid_avg, lags=lags, ax=ax)
            ax.set_title(f"ACF of Residual (Ljung-Box p={p_value:.4f})")
            fig.tight_layout()
            acf_path = os.path.join(image_dir, 'acf', f"{unique_id}.png")
            fig.savefig(acf_path)
            plt.close(fig)

        output_csv_path = os.path.join(feature_dir, f"{unique_id}.csv")
        pd.DataFrame([feature_dict]).to_csv(output_csv_path, index=False)
        logging.info(f"Features saved to {output_csv_path}")

    except Exception as e:
        logging.error(f"Error processing {unique_id}: {e}", exc_info=True)

if __name__ == "__main__":
    #unique_id = "112"
    unique_id = "3080"
    txt_path = f"/home/yanxunzhou/TBrain_SmartPing/data/raw/test_data/{unique_id}.txt"
    output_csv_dir = "/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/TimeSeriesFeatures/Test/features"
    output_img_dir = "/home/yanxunzhou/TBrain_SmartPing/tests/feature_eda/TimeSeriesFeatures/Test/images"
    generate_features(unique_id, txt_path, output_csv_dir, output_img_dir)
    print(f"Features saved to {output_csv_dir} and image saved to {output_img_dir}")
