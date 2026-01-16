import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import argparse
from datetime import datetime

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Get absolute path for models directory
MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Argument parsing
parser = argparse.ArgumentParser(description='Train cognitive load model (GPU) - by default uses your personalized feature set')
parser.add_argument('--columns', type=str, help='Comma-separated list of feature columns to use (exclude target). Overrides defaults if provided.')
parser.add_argument('--all-features', action='store_true', help='Use all available features (excluding target) instead of the personalized default.')
parser.add_argument('--save-selected-columns', type=str, help='Path to save the selected columns as a comma-separated txt file.')
args = parser.parse_args()

# Personalized feature set (the columns you provided, excluding target)
PERSONALIZED_FEATURES = [
    'Pupil_Dilation', 'Blink_Rate', 'Fixation_Duration', 'Saccade_Duration'
]

print("Loading cognitive load dataset...")
start_time = datetime.now()

# Load the dataset
df = pd.read_csv('Stressdetector\cognitive_load_dataset.csv')

# Show only personalized columns (that exist in the dataset)
personal_present = [c for c in PERSONALIZED_FEATURES if c in df.columns]
print(f"Dataset shape: {df.shape}")
print(f"Personalized columns present: {personal_present}")
if personal_present:
    print(f"\nFirst few rows (personalized columns):\n{df[personal_present].head()}")
else:
    print("\nNo personalized columns found in dataset; showing first few rows of full dataset:\n")
    print(df.head())

# Prepare data: selection precedence -> --columns > --all-features > personalized default
if args.columns:
    requested = [c.strip() for c in args.columns.split(',') if c.strip()]
    # Ensure requested columns are within the personalized feature set
    not_allowed = [c for c in requested if c not in PERSONALIZED_FEATURES]
    if not_allowed:
        raise ValueError(f"Columns not allowed. Only the personalized feature set may be used: {not_allowed}")
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    if not requested:
        raise ValueError("No valid columns specified in --columns")
    selected_cols = requested
    X = df[selected_cols].copy()
    print(f"\nUsing specified subset of personalized columns ({len(selected_cols)}): {selected_cols}")
elif args.all_features:
    X = df.drop('Cognitive_Load', axis=1)
    selected_cols = X.columns.tolist()
    print("\nUsing all feature columns (excluding 'Cognitive_Load').")
else:
    # Default to user's personalized feature list
    selected_cols = PERSONALIZED_FEATURES
    missing = [c for c in selected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Personalized default features missing from dataset: {missing}")
    X = df[selected_cols].copy()
    print(f"\nUsing personalized default features ({len(selected_cols)}): {selected_cols}")

y = df['Cognitive_Load']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle missing values (on selected features)
X = X.fillna(X.mean())

# Optionally save selected columns
if args.save_selected_columns:
    with open(args.save_selected_columns, 'w') as f:
        f.write(','.join(selected_cols))
    print(f"Saved selected columns to: {args.save_selected_columns}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("GPU-ACCELERATED MODEL TRAINING")
print("="*60)

# ============== XGBoost with GPU ==============
print("\n1. Training XGBoost with GPU acceleration...")
xgb_start = datetime.now()

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist',  # Histogram-based method
    device='cuda:0',  # Use GPU device
    n_jobs=-1,
    verbosity=1
)

xgb_model.fit(X_train_scaled, y_train)
xgb_time = datetime.now() - xgb_start

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluation
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f"   Training time: {xgb_time.total_seconds():.2f} seconds")
print(f"   MSE: {xgb_mse:.4f}")
print(f"   RMSE: {xgb_rmse:.4f}")
print(f"   MAE: {xgb_mae:.4f}")
print(f"   R² Score: {xgb_r2:.4f}")

# ============== LightGBM with GPU ==============
print("\n2. Training LightGBM with GPU acceleration...")
lgb_start = datetime.now()

lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    device='gpu',  # GPU acceleration
    gpu_platform_id=0,
    gpu_device_id=0,
    n_jobs=-1,
    verbosity=1
)

lgb_model.fit(X_train_scaled, y_train)
lgb_time = datetime.now() - lgb_start

# Predictions
y_pred_lgb = lgb_model.predict(X_test_scaled)

# Evaluation
lgb_mse = mean_squared_error(y_test, y_pred_lgb)
lgb_rmse = np.sqrt(lgb_mse)
lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
lgb_r2 = r2_score(y_test, y_pred_lgb)

print(f"   Training time: {lgb_time.total_seconds():.2f} seconds")
print(f"   MSE: {lgb_mse:.4f}")
print(f"   RMSE: {lgb_rmse:.4f}")
print(f"   MAE: {lgb_mae:.4f}")
print(f"   R² Score: {lgb_r2:.4f}")

# ============== Model Selection ==============
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

models_comparison = {
    'XGBoost': {'time': xgb_time.total_seconds(), 'r2': xgb_r2, 'rmse': xgb_rmse},
    'LightGBM': {'time': lgb_time.total_seconds(), 'r2': lgb_r2, 'rmse': lgb_rmse}
}

for model_name, metrics in models_comparison.items():
    print(f"\n{model_name}:")
    print(f"  Training Time: {metrics['time']:.2f}s")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")

# Select best model based on R² score
best_model_name = max(models_comparison, key=lambda x: models_comparison[x]['r2'])
print(f"\n✓ Best Model: {best_model_name}")

if best_model_name == 'XGBoost':
    best_model = xgb_model
    best_model_path = os.path.join(MODELS_DIR, 'cogload_model_xgb_gpu.joblib')
else:
    best_model = lgb_model
    best_model_path = os.path.join(MODELS_DIR, 'cogload_model_lgb_gpu.joblib')

# Save best model
joblib.dump(best_model, best_model_path)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'cogload_scaler.joblib'))

print(f"\n✓ Best model saved to: {best_model_path}")
print(f"✓ Scaler saved to: Stressdetector/models/cogload_scaler.joblib")

total_time = datetime.now() - start_time
print(f"\nTotal training time: {total_time.total_seconds():.2f} seconds")
print("="*60)
