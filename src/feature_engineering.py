"""
Feature Engineering Module
Creates features for time-series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import List


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features based on the datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Basic temporal features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding for temporal features.
    Cyclical encoding prevents the model from thinking 23:00 and 00:00 are far apart.
    
    Args:
        df: DataFrame with temporal features
        
    Returns:
        DataFrame with cyclical features
    """
    df = df.copy()
    
    # Hour cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def add_lag_features(df: pd.DataFrame, 
                     target_col: str = 'load',
                     lags: List[int] = None) -> pd.DataFrame:
    """
    Add lag features for time-series prediction.
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags from
        lags: List of lag periods (in hours)
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 24*7]  # 1h, 2h, 3h, 6h, 12h, 1d, 1w
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame,
                         target_col: str = 'load',
                         windows: List[int] = None) -> pd.DataFrame:
    """
    Add rolling statistics features.
    
    Args:
        df: Input DataFrame
        target_col: Column to compute rolling stats from
        windows: List of window sizes (in hours)
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    if windows is None:
        windows = [6, 12, 24, 24*7]  # 6h, 12h, 1d, 1w
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(window=window).max()
    
    return df


def add_diff_features(df: pd.DataFrame, target_col: str = 'load') -> pd.DataFrame:
    """
    Add difference features to capture trends.
    
    Args:
        df: Input DataFrame
        target_col: Column to compute differences from
        
    Returns:
        DataFrame with difference features
    """
    df = df.copy()
    
    # Hour-over-hour change
    df[f'{target_col}_diff_1h'] = df[target_col].diff(1)
    
    # Day-over-day change (same hour)
    df[f'{target_col}_diff_24h'] = df[target_col].diff(24)
    
    # Week-over-week change (same hour, same day)
    df[f'{target_col}_diff_168h'] = df[target_col].diff(24 * 7)
    
    return df


def create_features(df: pd.DataFrame, target_col: str = 'load') -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame with datetime index
        target_col: Name of the target column
        
    Returns:
        DataFrame with all engineered features
    """
    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_diff_features(df, target_col)
    
    # Drop rows with NaN from lag/rolling features
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = 'load') -> List[str]:
    """
    Get list of feature column names (excluding target).
    
    Args:
        df: DataFrame with features
        target_col: Name of target column to exclude
        
    Returns:
        List of feature column names
    """
    return [col for col in df.columns if col != target_col]


if __name__ == "__main__":
    # Example usage with sample data
    dates = pd.date_range(start='2024-01-01', periods=24*30, freq='H')
    sample_df = pd.DataFrame({
        'load': np.random.normal(1000, 200, len(dates)) + 
                100 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    }, index=dates)
    
    featured_df = create_features(sample_df, 'load')
    
    print(f"Original columns: {list(sample_df.columns)}")
    print(f"Feature columns: {len(get_feature_columns(featured_df))} features")
    print(f"\nSample features: {get_feature_columns(featured_df)[:10]}")
