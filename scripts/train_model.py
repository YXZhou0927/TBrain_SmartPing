import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing
from typing import Union, List, Dict, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import one_click

def run_one_click(method_name: str):
    """
    Quickly train, evaluate and predict using the specified method.
    Args:
        method_name (str): The name of the method to use for training.
    """
    # Check if the method name is provided
    if not method_name:
        raise ValueError("Method name must be provided.")
    # Check if the method name is a string
    if not isinstance(method_name, str):
        raise ValueError("Method name must be a string.")
    # Check if the method name is not empty
    if not method_name.strip():
        raise ValueError("Method name cannot be empty.")

    # Define the path to the training data
    train_info_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "train_info.csv"
    train_feature_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "train_features"
    test_info_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "test_info.csv"
    test_feature_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "test_features"
    sample_submission_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_submission.csv"

    
    # Perform training based on the method name
    if method_name == "Baseline":
        one_click.Baseline.run(train_info_path, train_feature_dir, test_info_path, test_feature_dir, sample_submission_path)
    elif method_name == "method2":
        # Training logic for method2
        pass

    print(f"Training completed using {method_name} method.")
    print("Output submission file to outputs/submissions directory.")

if __name__ == "__main__":
    # Example usage
    run_one_click("Baseline")