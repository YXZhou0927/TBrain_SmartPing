import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing
from typing import Union, List, Dict, Tuple


def load_data(meta_path: Union[str, Path], data_folder: Union[str, Path] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the metadata and data files.

    Args:
        meta_path (Union[str, Path]): Path to the metadata CSV file.
        data_folder (Union[str, Path], optional): Path to the folder containing data files. If None, only metadata is returned.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: A tuple containing the metadata DataFrame and a dictionary of file paths.
    """
    # Read metadata
    df = pd.read_csv(str(meta_path))
    df = df.drop(columns=['cut_point'], errors='ignore')  # Remove unnecessary columns
    df['unique_id'] = df['unique_id'].astype(str)  # Ensure unique_id is a string
    uid = df['unique_id'].tolist()  # List of unique IDs

    if data_folder is not None:
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder {data_folder} does not exist.")
        
        # Get list of data files
        data_files = list(data_folder.glob('**/*.txt'))
        data_files = [str(file) for file in data_files]  # Convert Path objects to strings
        data_files_dict = {Path(file).stem: file for file in data_files}  # Dictionary of unique_id to file path
        data_files_dict = {k: v for k, v in data_files_dict.items() if k in uid}  # Filter to only include files in metadata
        return df, data_files_dict
    else:
        return df, {}

def merge_metadata_and_features(meta_path: Union[str, Path], feature_folder: Union[str, Path]) -> pd.DataFrame:
    """
    Merge metadata DataFrame with corresponding feature CSV files.

    Args:
        meta_path (Union[str, Path]): Path to the metadata CSV file containing 'unique_id' column. 
        feature_folder (Union[str, Path]): Path to the folder containing feature CSV files named {unique_id}.csv.

    Returns:
        pd.DataFrame: Combined DataFrame with features columns prefixed by 'feature_'.
    """
    from tqdm import tqdm
    
    meta_df = load_data(meta_path)[0]  # Load metadata DataFrame
    feature_folder = Path(feature_folder)
    merged_rows = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Merging features"):
        uid = row['unique_id']
        feature_path = feature_folder / f"{uid}.csv"

        if feature_path.exists():
            try:
                feature_df = pd.read_csv(feature_path)
                feature_df.columns = [f"feature_{col}" for col in feature_df.columns]
                feature_df.insert(0, 'unique_id', uid)
                combined = pd.merge(row.to_frame().T, feature_df, on='unique_id', how='left')
                merged_rows.append(combined)
            except Exception as e:
                print(f"Error reading features for {uid}: {e}")

    return pd.concat(merged_rows, ignore_index=True) if merged_rows else meta_df

if __name__ == "__main__":
    # Example usage
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent # Assuming the script is in src/data
    meta_path = root_dir / "data" / "raw" / "train_info.csv"  # Path to metadata CSV file
    feature_folder = root_dir / "data" / "processed" / "train_features"  # Path to feature files
    
    # # Load metadata and data files
    # metadata, data_files = load_data(meta_path, data_folder)
    # print("Metadata loaded:", metadata.head())
    # print("Data files:", data_files)

    # Merge metadata with feature files
    merged_df = merge_metadata_and_features(meta_path, feature_folder)
    print("Merged DataFrame:", merged_df.head())