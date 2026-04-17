"""
Export ML predictions to CSV for Power BI

This script loads the trained model, generates predictions,
calculates metrics, and exports to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import joblib
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data'
MODELS_DIR = REPO_ROOT / 'models'
OUTPUT_FILE = DATA_DIR / 'california_housing_predictions.csv'

# Create data directory
DATA_DIR.mkdir(exist_ok=True)

# Load dataset
print("Loading California Housing dataset...")
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target
print(f"Loaded {len(df):,} properties")

# Load model
print("Loading trained model...")
model_path = MODELS_DIR / 'final_model.pkl'

if not model_path.exists():
    print("Error: Model not found at", model_path)
    print("Run notebooks/model_training.ipynb first")
    exit(1)

model = joblib.load(model_path)
print("Model loaded successfully")

# Generate predictions
print("Generating predictions...")
X = df.drop('PRICE', axis=1)
df['PREDICTED_PRICE'] = model.predict(X)

# Calculate error metrics
df['PRICE_ERROR'] = df['PRICE'] - df['PREDICTED_PRICE']
df['ERROR_PERCENT'] = (df['PRICE_ERROR'] / df['PRICE'] * 100).round(2)
df['ABS_ERROR_PERCENT'] = df['ERROR_PERCENT'].abs()

# Add geographic regions
def get_region(lat, lon):
    if lat > 37.5 and lon < -122.5:
        return 'SF Bay Area'
    elif lat > 36 and lon < -121:
        return 'Central Coast'
    elif lat < 34.5:
        return 'Southern California'
    else:
        return 'Other'

df['REGION'] = df.apply(lambda row: get_region(row['Latitude'], row['Longitude']), axis=1)

# Add income quartiles
df['INCOME_QUARTILE'] = pd.qcut(df['MedInc'], q=4, 
                                labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])

# Add price categories
def categorize_price(price):
    if price < 1.0:
        return 'Budget (<$100k)'
    elif price < 2.0:
        return 'Affordable ($100-200k)'
    elif price < 3.0:
        return 'Mid-range ($200-300k)'
    else:
        return 'Premium (>$300k)'

df['PRICE_CATEGORY'] = df['PRICE'].apply(categorize_price)

# Add accuracy categories
def categorize_accuracy(error_pct):
    abs_error = abs(error_pct)
    if abs_error < 5:
        return 'Excellent (<5%)'
    elif abs_error < 10:
        return 'Good (5-10%)'
    elif abs_error < 20:
        return 'Fair (10-20%)'
    else:
        return 'Poor (>20%)'

df['PREDICTION_ACCURACY'] = df['ERROR_PERCENT'].apply(categorize_accuracy)

# Export to CSV
df.to_csv(OUTPUT_FILE, index=False)

# Print results
print("\nData Export Summary")
print("=" * 50)
print(f"Output file: {OUTPUT_FILE}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print("\nError Metrics")
print("=" * 50)
print(f"Mean Error: ${df['PRICE_ERROR'].mean():.3f}k")
print(f"Mean Error %: {df['ERROR_PERCENT'].abs().mean():.2f}%")
print(f"Predictions < 5% error: {(df['ABS_ERROR_PERCENT'] < 5).sum() / len(df) * 100:.1f}%")
print(f"Predictions < 10% error: {(df['ABS_ERROR_PERCENT'] < 10).sum() / len(df) * 100:.1f}%")
print(f"Predictions < 20% error: {(df['ABS_ERROR_PERCENT'] < 20).sum() / len(df) * 100:.1f}%")
print("\nRegion Distribution")
print("=" * 50)
print(df['REGION'].value_counts().to_string())
print("\nPrice Category Distribution")
print("=" * 50)
print(df['PRICE_CATEGORY'].value_counts().to_string())
print("\nReady for Power BI!")