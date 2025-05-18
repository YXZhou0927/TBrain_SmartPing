import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Union, List, Dict, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data import load_data
from features import timeseries_features

def process_single_uid(args: Tuple[str, str, str]):
    uid, file_path, output_path = args
    timeseries_features.generate_features(uid, file_path, output_path, None)


def process_data(meta_path: Union[str, Path], data_folder: Union[str, Path], output_path: Union[str, Path]):
    """
    Process the data by loading metadata and data files, generating features, and saving the results.

    Args:
        meta_path (Union[str, Path]): Path to the metadata CSV file.
        data_folder (Union[str, Path]): Path to the folder containing data files.
        output_path (Union[str, Path]): Path to save the processed data.
    """
    # Load metadata and data files
    df, txt_files_dict = load_data(meta_path, data_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Prepare arguments for multiprocessing
    tasks = []
    for uid in df['unique_id']:
        uid_str = str(uid)
        if uid_str in txt_files_dict:
            file_path = txt_files_dict[uid_str]
            tasks.append((uid_str, file_path, str(output_path)))

    # Use multiprocessing pool to parallelize
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_uid, tasks), total=len(tasks), desc="Processing", ncols=80))

if __name__ == "__main__":
    # Example usage
    root_dir = Path(__file__).resolve().parent.parent
    for mode in ["train", "test"]:
        meta_path = root_dir / "data" / "raw" / f"{mode}_info.csv"
        data_folder = root_dir / "data" / "raw" / f"{mode}_data"
        output_path = root_dir / "data" / "processed" / f"{mode}_features"

        process_data(meta_path, data_folder, output_path)
        print(f"Processed {mode} data and saved to {output_path}")
    print("All data processed successfully.")