"""
Energy Demand Forecasting Pipeline
Main entry point for running the forecasting system.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_pipeline, load_data, handle_missing_values
from src.feature_engineering import create_features
from src.models import train_models
from src.evaluation import evaluate_models, generate_report


def generate_sample_data(filepath: str, n_days: int = 60) -> None:
    """
    Generate sample electricity load data for demonstration.
    
    Args:
        filepath: Path to save the sample data
        n_days: Number of days of data to generate
    """
    np.random.seed(42)
    
    # Generate hourly timestamps
    dates = pd.date_range(start='2024-01-01', periods=24 * n_days, freq='H')
    
    # Base load pattern
    base_load = 1000
    
    # Daily pattern (higher during day, lower at night)
    hour_effect = 200 * np.sin((np.arange(24 * n_days) - 6) * np.pi / 12)
    
    # Weekly pattern (lower on weekends)
    day_of_week = dates.dayofweek
    weekend_effect = -100 * (day_of_week >= 5).astype(int)
    
    # Seasonal pattern
    month = dates.month
    seasonal_effect = 150 * np.sin((month - 1) * np.pi / 6)
    
    # Random noise
    noise = np.random.normal(0, 50, len(dates))
    
    # Combine all effects
    load = base_load + hour_effect + np.tile(weekend_effect[:24*7], n_days // 7 + 1)[:len(dates)] + seasonal_effect + noise
    
    # Ensure positive values
    load = np.maximum(load, 100)
    
    # Add some missing values for realism
    missing_idx = np.random.choice(len(dates), size=int(len(dates) * 0.002), replace=False)
    load[missing_idx] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'load': load
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Generated sample data: {len(df)} records saved to {filepath}")


def run_pipeline(data_path: str, output_dir: str = 'output') -> None:
    """
    Run the complete forecasting pipeline.
    
    Args:
        data_path: Path to input data CSV
        output_dir: Directory to save outputs
    """
    print("=" * 60)
    print("Energy Demand Forecasting Pipeline")
    print("=" * 60)
    
    # Check if sample data exists, if not generate it
    if not os.path.exists(data_path):
        print(f"\nData file not found. Generating sample data...")
        generate_sample_data(data_path)
    
    # Load and preprocess data
    print("\nLoading data...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    print(f"✓ Loaded {len(df)} hourly records")
    
    # Handle missing values
    print("\nPreprocessing...")
    missing_count = df['load'].isna().sum()
    df['load'] = df['load'].interpolate(method='time').bfill().ffill()
    print(f"✓ Handled {missing_count} missing values")
    
    # Detect outliers (simple IQR method)
    Q1 = df['load'].quantile(0.25)
    Q3 = df['load'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df['load'] < Q1 - 1.5 * IQR) | (df['load'] > Q3 + 1.5 * IQR)
    outlier_count = outlier_mask.sum()
    df = df[~outlier_mask]
    print(f"✓ Removed {outlier_count} outliers")
    
    # Feature engineering
    print("\nFeature Engineering...")
    df_featured = create_features(df, 'load')
    feature_count = len(df_featured.columns) - 1
    print(f"✓ Created {feature_count} features (temporal, lag, rolling)")
    
    # Train models
    print("\nTraining Models...")
    results = train_models(df_featured, 'load')
    
    # Evaluate
    metrics_df = evaluate_models(results)
    
    for _, row in metrics_df.iterrows():
        print(f"✓ {row['Model']}: MAE = {row['MAE']:.1f} MW")
    
    # Generate outputs
    print("\nGenerating Forecasts...")
    generate_report(results, output_dir)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'timestamp': results['y_test'].index,
        'actual': results['y_test'].values,
        'predicted': results['gradient_boosting']['predictions']
    })
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ 24-hour ahead predictions saved to {predictions_path}")
    print(f"✓ Visualizations saved to {output_dir}/plots/")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Energy Demand Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with sample data
  python main.py --data my_data.csv       # Run with custom data
  python main.py --output results/        # Custom output directory
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_load_data.csv',
        help='Path to input CSV file with timestamp and load columns'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory to save outputs (predictions, plots, metrics)'
    )
    
    args = parser.parse_args()
    
    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
