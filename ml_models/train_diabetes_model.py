import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflowjs as tfjs
import os

print("=" * 60)
print("DIABETES PREDICTION MODEL TRAINING")
print("=" * 60)

print("\n1. Loading Pima Indians Diabetes Dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

try:
    data = pd.read_csv(url, names=column_names)
    print(f"✓ Dataset loaded: {data.shape[0]} samples, {data.shape[1]-1} features")
    print(f"✓ Positive cases (diabetes): {data['Outcome'].sum()} ({data['Outcome'].mean()*100:.1f}%)")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("Please check your internet connection")
    exit(1)

print("\n2. Preprocessing data...")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_columns] = data[zero_columns].replace(0, np.nan)

data.fillna(data.median(), inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

print(f"✓ Features: {list(X.columns)}")
print(f"✓ Missing values handled")
print("\n3. Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")


print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features normalized using StandardScaler")

scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': list(X.columns)
}


print("\n5. Building Neural Network model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

print("\n6. Training model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
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
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Save model for TensorFlow.js
print("\n8. Exporting model to TensorFlow.js format...")
output_dir = './diabetes_model_tfjs'
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
    'dataset': 'Pima Indians Diabetes Dataset',
    'samples': int(data.shape[0]),
    'features': list(X.columns),
    'test_accuracy': float(test_accuracy),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_path = os.path.join(output_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Model metadata saved to: {metadata_path}")

print("\n" + "=" * 60)
print("✓ DIABETES MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel accuracy: {test_accuracy*100:.2f}%")
print(f"Output directory: {output_dir}")
print("\nNext steps:")
print("1. Copy the 'diabetes_model_tfjs' folder to your web project")
print("2. Update HTML to load this model using TensorFlow.js")
print("=" * 60)
