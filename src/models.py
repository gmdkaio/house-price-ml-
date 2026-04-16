import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_models():
    """Create all models to train"""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,              # Changed from 15
            min_samples_split=10,      # Added 
            min_samples_leaf=5,        # Added
            max_features='log2',       # Added
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and return evaluation results"""
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Store results
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }
        
        print(f"{name} trained")
    
    return results

def compare_models(results):
    """Create comparison dataframe"""
    comparison = []
    
    for name, metrics in results.items():
        comparison.append({
            'Model': name,
            'Train R2': metrics['train_r2'],
            'Test R2': metrics['test_r2'],
            'Train MAE': metrics['train_mae'],
            'Test MAE': metrics['test_mae'],
            'Train RMSE': metrics['train_rmse'],
            'Test RMSE': metrics['test_rmse'],
            'Overfit': metrics['train_r2'] - metrics['test_r2']
        })
    
    return pd.DataFrame(comparison)

def get_best_model(results, comparison_df):
    """Get the best performing model"""
    best_model_name = comparison_df.loc[comparison_df['Test R2'].idxmax(), 'Model']
    best_model_info = results[best_model_name]
    
    return best_model_name, best_model_info