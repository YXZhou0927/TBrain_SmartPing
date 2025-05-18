import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks

def perform_fft(signal, sampling_rate):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)
    magnitude = np.abs(fft_result)
    half = len(freq) // 2
    return freq[:half], magnitude[:half], fft_result

def bandpass_ifft_feature(fft_result, freq, center_idx, lower_ratio, upper_ratio):
    N = len(fft_result)
    mask = np.zeros_like(fft_result, dtype=complex)

    # 判斷保留區間的範圍
    lower_bound = int(center_idx * lower_ratio)
    upper_bound = int(center_idx * upper_ratio)

    # 保留正頻率對應區間
    mask[lower_bound:upper_bound + 1] = fft_result[lower_bound:upper_bound + 1]

    # 保留對稱的負頻率區間
    if lower_bound > 0:
        mask[-upper_bound:-lower_bound + 1] = fft_result[-upper_bound:-lower_bound + 1]

    # IFFT 並取絕對值
    feature_signal = np.real(np.fft.ifft(mask))
    return feature_signal

class FFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("6-Axis Sensor FFT Viewer")

        self.data = None
        self.txt_path = None

        # === File selection ===
        self.btn_select = tk.Button(root, text="Select TXT", command=self.select_txt)
        self.btn_select.grid(row=0, column=0, padx=5, pady=5)

        self.label_info = tk.Label(root, text="No file selected", anchor="w", width=40)
        self.label_info.grid(row=0, column=1, sticky="w")

        # === Sampling Rate ===
        tk.Label(root, text="Sampling Rate (Hz):").grid(row=1, column=0, sticky="e", padx=5)
        self.entry_sampling = tk.Entry(root)
        self.entry_sampling.insert(0, "85")
        self.entry_sampling.grid(row=1, column=1, sticky="w")

        # === Lower/Upper Frequency Ratios ===
        tk.Label(root, text="Lower bound ratio:").grid(row=2, column=0, sticky="e", padx=5)
        self.entry_lower = tk.Entry(root)
        self.entry_lower.insert(0, "0.8")
        self.entry_lower.grid(row=2, column=1, sticky="w")

        tk.Label(root, text="Upper bound ratio:").grid(row=3, column=0, sticky="e", padx=5)
        self.entry_upper = tk.Entry(root)
        self.entry_upper.insert(0, "1.2")
        self.entry_upper.grid(row=3, column=1, sticky="w")

        # === Start Button ===
        self.btn_convert = tk.Button(root, text="Start Analysis", command=self.analyze_data, state="disabled")
        self.btn_convert.grid(row=4, column=0, columnspan=2, pady=10)

    def select_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            data = np.loadtxt(path)
            if data.shape[1] != 6:
                raise ValueError("Expected 6 columns (ax, ay, az, gx, gy, gz)")
            self.data = data
            self.txt_path = path
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            filename = path.split("/")[-1]
            self.label_info.config(text=f"{filename} loaded at {timestamp}")
            self.btn_convert.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load TXT file:\n{e}")

    def analyze_data(self):
        try:
            sampling_rate = float(self.entry_sampling.get())
            lower_ratio = float(self.entry_lower.get())
            upper_ratio = float(self.entry_upper.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sampling rate or filter ratios.")
            return

        ax, ay, az, gx, gy, gz = self.data.T
        a = np.sqrt(ax**2 + ay**2 + az**2)
        g = np.sqrt(gx**2 + gy**2 + gz**2)
        
        ### Testing ###
        #a = np.concatenate((np.full(int(len(a)/2), np.mean(a)), a, np.full(int(len(a)/2), np.mean(a))))
        #g = np.concatenate((np.full(int(len(g)/2), np.mean(g)), g, np.full(int(len(g)/2), np.mean(g))))
        
        a_energy = np.sum(a**2) / len(a)
        g_energy = np.sum(g**2) / len(g)
        c = np.sqrt(a**2 / a_energy + g**2 / g_energy)
        c = np.convolve(c, np.ones(50)/50, mode='same')
        time = np.arange(len(a)) / sampling_rate

        a_demeaned = a - np.mean(a)
        g_demeaned = g - np.mean(g)
        c_demeaned = c - np.mean(c)
        freq_a, mag_a, fft_a = perform_fft(a_demeaned, sampling_rate)
        freq_g, mag_g, fft_g = perform_fft(g_demeaned, sampling_rate)
        freq_c, mag_c, fft_c = perform_fft(c_demeaned, sampling_rate)

        # 主頻率 index
        lower_idx = np.where(1.0 / freq_c[1:] > 1.2)[0][-1] + 1
        upper_idx = np.where(1.0 / freq_c[1:] < 0.7)[0][0] + 1
        idx_a = np.argmax(mag_a[lower_idx:upper_idx] * freq_a[lower_idx:upper_idx]**2) + lower_idx
        idx_g = np.argmax(mag_g[lower_idx:upper_idx] * freq_g[lower_idx:upper_idx]**2) + lower_idx
        idx_c = np.argmax(mag_c[lower_idx:upper_idx] * freq_c[lower_idx:upper_idx]**2) + lower_idx
        dom_freq_a = freq_a[idx_a]
        dom_freq_g = freq_g[idx_g]
        dom_freq_c = freq_c[idx_c]
        dom_period_a = 1.0 / dom_freq_a if dom_freq_a != 0 else np.inf
        dom_period_g = 1.0 / dom_freq_g if dom_freq_g != 0 else np.inf
        dom_period_c = 1.0 / dom_freq_c if dom_freq_c != 0 else np.inf
        duration = len(a) / sampling_rate
        repeats_a = duration / dom_period_a if dom_period_a != np.inf else 0
        repeats_g = duration / dom_period_g if dom_period_g != np.inf else 0
        repeats_c = duration / dom_period_c if dom_period_c != np.inf else 0

        # 特徵訊號
        feature_a = bandpass_ifft_feature(fft_a, freq_a, idx_a, lower_ratio, upper_ratio)
        feature_g = bandpass_ifft_feature(fft_g, freq_g, idx_g, lower_ratio, upper_ratio)
        feature_c = bandpass_ifft_feature(fft_c, freq_c, idx_c, lower_ratio, upper_ratio)

        top_threshold = 0.5
        temp = feature_c / np.max(np.abs(feature_c))
        cut_points, _ = find_peaks(temp, height=top_threshold)
        print(f"Cut points: {cut_points}")
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        ax1.plot(time, a_demeaned / np.max(np.abs(a)), label="Accel |a| (demeaned)")
        ax1.plot(time, g_demeaned / np.max(np.abs(g)), label="Gyro |g| (demeaned)")
        ax1.plot(time, c_demeaned / np.max(np.abs(c)), label="Combined |c| (demeaned)")
        ax1.plot(time, feature_c / np.max(np.abs(feature_c)), label="Feature |c| (demeaned)", linestyle="--")
        ax1.set_title("Raw Signals and Feature Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Magnitude")
        ax1.legend()

        ax2.plot(freq_a, mag_a, label="Accel FFT")
        ax2.plot(freq_g, mag_g, label="Gyro FFT")
        ax2.set_title(
            "FFT Spectrum\n"
            f"Accel: {dom_freq_a:.3f} Hz | T={dom_period_a:.2f}s | ~{repeats_a:.1f} repeats\n"
            f"Gyro:  {dom_freq_g:.3f} Hz | T={dom_period_g:.2f}s | ~{repeats_g:.1f} repeats\n"
            f"Combined: {dom_freq_c:.3f} Hz | T={dom_period_c:.2f}s | ~{repeats_c:.1f} repeats\n"
        )
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = FFTApp(root)
    root.mainloop()
