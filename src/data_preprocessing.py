"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load electricity load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with datetime index and load column
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> Tuple[pd.DataFrame, int]:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: Method to handle missing values ('interpolate', 'forward_fill', 'drop')
        
    Returns:
        Tuple of (cleaned DataFrame, count of missing values handled)
    """
    missing_count = df['load'].isna().sum()
    
    if method == 'interpolate':
        df['load'] = df['load'].interpolate(method='time')
    elif method == 'forward_fill':
        df['load'] = df['load'].ffill()
    elif method == 'drop':
        df = df.dropna(subset=['load'])
    
    # Handle any remaining NaN at edges
    df['load'] = df['load'].bfill().ffill()
    
    return df, missing_count


def detect_and_remove_outliers(df: pd.DataFrame, 
                                column: str = 'load',
                                method: str = 'iqr',
                                threshold: float = 1.5) -> Tuple[pd.DataFrame, int]:
    """
    Detect and remove outliers from the dataset.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (cleaned DataFrame, count of outliers removed)
    """
    original_count = len(df)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        df = df[mask]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]
    
    outliers_removed = original_count - len(df)
    return df, outliers_removed


def preprocess_pipeline(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        filepath: Path to the raw data file
        
    Returns:
        Tuple of (preprocessed DataFrame, statistics dict)
    """
    stats = {}
    
    # Load data
    df = load_data(filepath)
    stats['original_rows'] = len(df)
    
    # Handle missing values
    df, missing_count = handle_missing_values(df)
    stats['missing_handled'] = missing_count
    
    # Remove outliers
    df, outliers_count = detect_and_remove_outliers(df)
    stats['outliers_removed'] = outliers_count
    
    stats['final_rows'] = len(df)
    
    return df, stats


if __name__ == "__main__":
    # Example usage
    import os
    
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_load_data.csv')
    
    if os.path.exists(sample_path):
        df, stats = preprocess_pipeline(sample_path)
        print(f"Preprocessing complete:")
        print(f"  Original rows: {stats['original_rows']}")
        print(f"  Missing handled: {stats['missing_handled']}")
        print(f"  Outliers removed: {stats['outliers_removed']}")
        print(f"  Final rows: {stats['final_rows']}")
    else:
        print(f"Sample data not found at {sample_path}")
