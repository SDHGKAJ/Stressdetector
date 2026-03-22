"""
Train a machine learning model for cognitive load detection
using EEG and vehicle sensor data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import os

# Load the cognitive load dataset
print("Loading cognitive load dataset...")
dataset_path = 'Stressdetector\cognitive_load_dataset.csv'
df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[-10:]}")
print(f"\nCognitive Load distribution:")
print(df['Cognitive_Load'].value_counts().sort_index())

# Separate features and target
X = df.drop('Cognitive_Load', axis=1)
y = df['Cognitive_Load']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Convert target to classification problem (0=Low, 1=Medium, 2=High)
# Assuming the values are already 0, 1, 2 for low, medium, high cognitive load
y = y.astype(int)

# Handle any missing values
X = X.fillna(X.mean())

# Split the data: 80% training, 20% testing
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and compare
print("\n" + "="*60)
print("Training Models...")
print("="*60)

# Model 1: Random Forest (Optimized for speed)
print("\n1. Random Forest Classifier")
print("-" * 40)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

# Train on full training set (skip cross-val for speed)
print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
print(f"Test Accuracy: {rf_accuracy:.4f}")
print(f"Test F1 (macro): {rf_f1:.4f}")

# Model 2: Gradient Boosting (Optimized for speed)
print("\n2. Gradient Boosting Classifier")
print("-" * 40)
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Train
print("Training Gradient Boosting...")
gb_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_gb = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb, average='macro')
print(f"Test Accuracy: {gb_accuracy:.4f}")
print(f"Test F1 (macro): {gb_f1:.4f}")

# Select best model
print("\n" + "="*60)
print("Model Comparison")
print("="*60)
print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")
print(f"Gradient Boosting - Accuracy: {gb_accuracy:.4f}, F1: {gb_f1:.4f}")

if gb_f1 >= rf_f1:
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_pred = y_pred_gb
    best_accuracy = gb_accuracy
    best_f1 = gb_f1
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_pred = y_pred_rf
    best_accuracy = rf_accuracy
    best_f1 = rf_f1

print(f"\nBest Model: {best_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")

# Detailed evaluation
print("\n" + "="*60)
print(f"Detailed Results for {best_name}")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, best_pred, 
                          target_names=['Low', 'Medium', 'High'],
                          digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    print("\nTop 20 Most Important Features:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))

# Save the best model
os.makedirs('Stressdetector/models', exist_ok=True)
model_path = 'Stressdetector/models/cognitive_load_model.joblib'
joblib.dump(best_model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save the scaler
scaler_path = 'Stressdetector/models/cognitive_load_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to: {scaler_path}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
