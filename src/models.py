"""
Models Module
Training and prediction for energy demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BaselineModel:
    """
    Simple baseline model using moving average.
    Useful for benchmarking more complex models.
    """
    
    def __init__(self, window: int = 24 * 7):
        """
        Initialize baseline model.
        
        Args:
            window: Window size for moving average (default: 1 week)
        """
        self.window = window
        self.history = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineModel':
        """Store training data for reference."""
        self.history = y.copy()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using rolling mean of historical values.
        
        For simplicity, uses the most recent window mean as prediction.
        """
        if self.history is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use the mean of the last `window` observations
        recent_mean = self.history.tail(self.window).mean()
        return np.full(len(X), recent_mean)


class EnergyDemandModel:
    """
    Gradient Boosting model for energy demand forecasting.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model with optional hyperparameters.
        
        Args:
            **kwargs: Parameters passed to GradientBoostingRegressor
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 10,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = GradientBoostingRegressor(**default_params)
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnergyDemandModel':
        """
        Fit the model to training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Fitted model instance
        """
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure columns match training
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def train_test_split_temporal(df: pd.DataFrame, 
                               target_col: str = 'load',
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data temporally (no shuffling) to prevent data leakage.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


def train_models(df: pd.DataFrame, 
                 target_col: str = 'load') -> Dict[str, Any]:
    """
    Train all models and return fitted models with test data.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        Dictionary containing models, predictions, and test data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_temporal(df, target_col)
    
    # Train baseline model
    baseline = BaselineModel(window=24*7)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    
    # Train gradient boosting model
    gb_model = EnergyDemandModel()
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    return {
        'baseline': {
            'model': baseline,
            'predictions': baseline_pred
        },
        'gradient_boosting': {
            'model': gb_model,
            'predictions': gb_pred
        },
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    from feature_engineering import create_features
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=24*60, freq='H')
    sample_df = pd.DataFrame({
        'load': np.random.normal(1000, 200, len(dates)) + 
                150 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    }, index=dates)
    
    # Create features
    featured_df = create_features(sample_df, 'load')
    
    # Train models
    results = train_models(featured_df, 'load')
    
    print("Models trained successfully!")
    print(f"Training samples: {len(results['X_train'])}")
    print(f"Test samples: {len(results['X_test'])}")
    
    # Show feature importance
    importance = results['gradient_boosting']['model'].get_feature_importance()
    print(f"\nTop 5 features:")
    print(importance.head())
