"""Heart Disease Prediction Model Training Script
Uses UCI Heart Disease Dataset to train a Neural Network model
Exports to TensorFlow.js format for browser usage
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflowjs as tfjs
import os

print("=" * 60)
print("HEART DISEASE PREDICTION MODEL TRAINING")
print("=" * 60)

# Download dataset
print("\n1. Loading UCI Heart Disease Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

try:
    data = pd.read_csv(url, names=column_names, na_values='?')
    print(f"✓ Dataset loaded: {data.shape[0]} samples, {data.shape[1]-1} features")
    
    # Convert target to binary (0 = no disease, 1-4 = disease present)
    data['target'] = (data['target'] > 0).astype(int)
    print(f"✓ Positive cases (heart disease): {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("Please check your internet connection")
    exit(1)

# Data preprocessing
print("\n2. Preprocessing data...")
# Handle missing values
print(f"  Missing values before: {data.isnull().sum().sum()}")
data.dropna(inplace=True)
print(f"  Missing values after: {data.isnull().sum().sum()}")
print(f"  Remaining samples: {len(data)}")

# Separate features and target
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

# Save scaler parameters for JavaScript
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': list(X.columns)
}

# Build Neural Network model
print("\n5. Building Neural Network model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

# Train model
print("\n6. Training model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    verbose=0
)

print(f"✓ Training completed")
print(f"✓ Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"✓ Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# Evaluate model
print("\n7. Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")

# Predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Save model for TensorFlow.js
print("\n8. Exporting model to TensorFlow.js format...")
output_dir = './heart_disease_model_tfjs'
os.makedirs(output_dir, exist_ok=True)

tfjs.converters.save_keras_model(model, output_dir)
print(f"✓ Model saved to: {output_dir}")

# Save scaler parameters as JSON
import json
scaler_path = os.path.join(output_dir, 'scaler_params.json')
with open(scaler_path, 'w') as f:
    json.dump(scaler_params, f, indent=2)
print(f"✓ Scaler parameters saved to: {scaler_path}")

# Save model metadata
metadata = {
    'model_type': 'Neural Network',
    'dataset': 'UCI Heart Disease Dataset',
    'samples': int(len(data)),
    'features': list(X.columns),
    'test_accuracy': float(test_accuracy),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_path = os.path.join(output_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Model metadata saved to: {metadata_path}")

print("\n" + "=" * 60)
print("✓ HEART DISEASE MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel accuracy: {test_accuracy*100:.2f}%")
print(f"Output directory: {output_dir}")
print("\nNext steps:")
print("1. Copy the 'heart_disease_model_tfjs' folder to your web project")
print("2. Update HTML to load this model using TensorFlow.js")
print("=" * 60)
