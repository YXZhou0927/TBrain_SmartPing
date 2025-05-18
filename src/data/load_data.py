import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing
from typing import Union, List, Dict, Tuple


def load_data(meta_path: Union[str, Path], data_folder: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the metadata and data files.

    Args:
        meta_path (Union[str, Path]): Path to the metadata CSV file.
        data_folder (Union[str, Path]): Path to the folder containing data files.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: A tuple containing the metadata DataFrame and a dictionary of file paths.
    """
    # Read metadata
    df = pd.read_csv(str(meta_path))
    df = df.drop(columns=['cut_point'], errors='ignore')  # Remove unnecessary columns
    df['unique_id'] = df['unique_id'].astype(str)  # Ensure unique_id is a string
    uid = df['unique_id'].tolist()  # List of unique IDs

    # Get list of data files
    data_files = list(Path(data_folder).glob('**/*.txt'))
    data_files = [str(file) for file in data_files]  # Convert Path objects to strings
    data_files_dict = {Path(file).stem: file for file in data_files}  # Dictionary of unique_id to file path
    data_files_dict = {k: v for k, v in data_files_dict.items() if k in uid}  # Filter to only include files in metadata

    return df, data_files_dict
    