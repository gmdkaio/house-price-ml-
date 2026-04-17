"""
Final trained model for house price prediction.

Aggressive Random Forest with hyperparameter tuning to reduce overfitting.
Test R2: 0.7929 | Test MAE: $348k | Overfitting Gap: 8.1%
"""

import joblib
from pathlib import Path


def load_model():
    """Load the final trained model."""
    model_path = Path(__file__).parent / 'final_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def save_model(model):
    """Save the final trained model."""
    model_path = Path(__file__).parent / 'final_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")