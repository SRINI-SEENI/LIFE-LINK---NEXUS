"""
Simplified Heart Disease Prediction Model Training
Uses scikit-learn RandomForest (works with Python 3.12+)
Exports model and creates JavaScript-compatible format
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

print("=" * 60)
print("HEART DISEASE PREDICTION MODEL TRAINING (RandomForest)")
print("=" * 60)

# Download dataset
print("\n1. Loading UCI Heart Disease Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

try:
    data = pd.read_csv(url, names=column_names, na_values='?')
    print(f"✓ Dataset loaded: {data.shape[0]} samples, {data.shape[1]-1} features")
    
    # Convert target to binary
    data['target'] = (data['target'] > 0).astype(int)
    print(f"✓ Positive cases (heart disease): {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# Data preprocessing
print("\n2. Preprocessing data...")
print(f"  Missing values before: {data.isnull().sum().sum()}")
data.dropna(inplace=True)
print(f"  Missing values after: {data.isnull().sum().sum()}")
print(f"  Remaining samples: {len(data)}")

X = data.drop('target', axis=1)
y = data['target']

print(f"✓ Features: {list(X.columns)}")

# Split dataset
print("\n3. Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Feature scaling
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features normalized using StandardScaler")

# Build RandomForest model
print("\n5. Building RandomForest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train model
print("\n6. Training model...")
model.fit(X_train_scaled, y_train)
print("✓ Training completed")

# Evaluate model
print("\n7. Evaluating model on test set...")
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Feature importance
print("\nFeature Importance:")
feature_importance = dict(zip(X.columns, model.feature_importances_))
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")

# Save model
print("\n8. Saving model...")
output_dir = './heart_disease_model'
os.makedirs(output_dir, exist_ok=True)

# Save scikit-learn model
model_path = os.path.join(output_dir, 'heart_model.joblib')
joblib.dump(model, model_path)
print(f"✓ Model saved to: {model_path}")

# Save scaler
scaler_path = os.path.join(output_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to: {scaler_path}")

# Save parameters for JavaScript
js_model = {
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'feature_names': list(X.columns),
    'feature_importance': feature_importance,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'test_accuracy': float(test_accuracy),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'dataset': 'UCI Heart Disease Dataset',
    'total_samples': int(len(data))
}

js_path = os.path.join(output_dir, 'model_params.json')
with open(js_path, 'w') as f:
    json.dump(js_model, f, indent=2)
print(f"✓ Model parameters saved to: {js_path}")

print("\n" + "=" * 60)
print("✓ HEART DISEASE MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel accuracy: {test_accuracy*100:.2f}%")
print(f"Output directory: {output_dir}")
print("=" * 60)
