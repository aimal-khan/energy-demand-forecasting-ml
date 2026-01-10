"""
Evaluation Module
Metrics calculation and visualization for forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred)
    }


def evaluate_models(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Evaluate all models and return comparison DataFrame.
    
    Args:
        results: Dictionary from train_models function
        
    Returns:
        DataFrame comparing model performance
    """
    y_true = results['y_test'].values
    
    metrics_list = []
    
    for model_name in ['baseline', 'gradient_boosting']:
        y_pred = results[model_name]['predictions']
        metrics = calculate_all_metrics(y_true, y_pred)
        metrics['Model'] = model_name.replace('_', ' ').title()
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)[['Model', 'MAE', 'MAPE', 'RMSE']]


def plot_predictions(results: Dict[str, Any], 
                     model_name: str = 'gradient_boosting',
                     output_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        results: Dictionary from train_models function
        model_name: Which model's predictions to plot
        output_path: Path to save the plot (optional)
    """
    y_true = results['y_test']
    y_pred = results[model_name]['predictions']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(y_true.index, y_true.values, label='Actual', alpha=0.7)
    ax1.plot(y_true.index, y_pred, label='Predicted', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (MW)')
    ax1.set_title(f'Actual vs Predicted - {model_name.replace("_", " ").title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true.values, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Load (MW)')
    ax2.set_ylabel('Predicted Load (MW)')
    ax2.set_title('Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.close()


def plot_feature_importance(model: Any, 
                            top_n: int = 15,
                            output_path: Optional[str] = None) -> None:
    """
    Plot feature importance for the model.
    
    Args:
        model: Trained EnergyDemandModel
        top_n: Number of top features to show
        output_path: Path to save the plot
    """
    importance_df = model.get_feature_importance().head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.barplot(data=importance_df, y='feature', x='importance', ax=ax, palette='viridis')
    
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.close()


def plot_residuals(results: Dict[str, Any],
                   model_name: str = 'gradient_boosting',
                   output_path: Optional[str] = None) -> None:
    """
    Plot residual analysis.
    
    Args:
        results: Dictionary from train_models function
        model_name: Which model to analyze
        output_path: Path to save the plot
    """
    y_true = results['y_test'].values
    y_pred = results[model_name]['predictions']
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual distribution
    ax1 = axes[0]
    ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Residual Distribution')
    ax1.legend()
    
    # Residuals vs Predicted
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Value')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residuals vs Predicted')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.close()


def generate_report(results: Dict[str, Any], output_dir: str = 'output') -> None:
    """
    Generate complete evaluation report with all plots.
    
    Args:
        results: Dictionary from train_models function
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Evaluate models
    metrics_df = evaluate_models(results)
    print("\nModel Comparison:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    
    # Generate plots
    plot_predictions(results, 'gradient_boosting', 
                     os.path.join(plots_dir, 'predictions.png'))
    
    plot_feature_importance(results['gradient_boosting']['model'], 
                            output_path=os.path.join(plots_dir, 'feature_importance.png'))
    
    plot_residuals(results, 'gradient_boosting',
                   os.path.join(plots_dir, 'residuals.png'))
    
    print(f"\nReport generated in {output_dir}/")


if __name__ == "__main__":
    # Example with synthetic data
    from feature_engineering import create_features
    from models import train_models
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=24*60, freq='H')
    sample_df = pd.DataFrame({
        'load': np.random.normal(1000, 200, len(dates)) + 
                150 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    }, index=dates)
    
    # Create features and train
    featured_df = create_features(sample_df, 'load')
    results = train_models(featured_df, 'load')
    
    # Generate report
    generate_report(results, 'output')
