# Machine Learning Models Training Guide
# LifeLink Nexus - Real ML Implementation

## Step-by-Step Setup

### Step 1: Install Python Dependencies

Open PowerShell in the `ml_models` folder and run:

```powershell
cd "C:\Users\Work\Desktop\LNN\ml_models"
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.13.0
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- tensorflowjs 4.10.0

### Step 2: Train Diabetes Model

```powershell
python train_diabetes_model.py
```

**What this does:**
- Downloads Pima Indians Diabetes Dataset (768 samples)
- Trains Neural Network model (100 epochs)
- Achieves ~75-80% accuracy
- Exports model to `diabetes_model_tfjs/` folder
- Creates scaler parameters and metadata

**Output files:**
```
diabetes_model_tfjs/
├── model.json              # Model architecture
├── group1-shard1of1.bin    # Model weights
├── scaler_params.json      # Feature scaling parameters
└── model_metadata.json     # Training info
```

### Step 3: Train Heart Disease Model

```powershell
python train_heart_disease_model.py
```

**What this does:**
- Downloads UCI Heart Disease Dataset (303 samples)
- Trains Neural Network model (150 epochs)
- Achieves ~80-85% accuracy
- Exports model to `heart_disease_model_tfjs/` folder
- Creates scaler parameters and metadata

**Output files:**
```
heart_disease_model_tfjs/
├── model.json              # Model architecture
├── group1-shard1of1.bin    # Model weights
├── scaler_params.json      # Feature scaling parameters
└── model_metadata.json     # Training info
```

### Step 4: Copy Models to Web Project

After training, copy the model folders:

```powershell
Copy-Item -Path "diabetes_model_tfjs" -Destination "..\user\models\" -Recurse

Copy-Item -Path "heart_disease_model_tfjs" -Destination "..\user\models\" -Recurse
```

## Model Details

### Diabetes Model
- **Architecture**: Neural Network (4 layers)
  - Input: 8 features
  - Hidden: 16 → 12 → 8 neurons (ReLU activation)
  - Output: 1 neuron (Sigmoid activation)
- **Features**: Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, Pedigree, Age
- **Dataset**: Pima Indians Diabetes (768 samples)
- **Expected Accuracy**: 75-80%

### Heart Disease Model
- **Architecture**: Neural Network (4 layers)
  - Input: 13 features
  - Hidden: 20 → 16 → 12 neurons (ReLU activation)
  - Output: 1 neuron (Sigmoid activation)
- **Features**: Age, Sex, Chest Pain Type, BP, Cholesterol, etc.
- **Dataset**: UCI Heart Disease (303 samples)
- **Expected Accuracy**: 80-85%

## 🔧 Troubleshooting

### Issue: "No module named tensorflow"
```powershell
pip install --upgrade pip
pip install tensorflow==2.13.0
```

### Issue: TensorFlow installation fails
Try CPU-only version:
```powershell
pip install tensorflow-cpu==2.13.0
```

### Issue: Dataset download fails
- Check internet connection
- Try running script again
- Datasets are public and freely available

## Verification

After training, you should see:

✓ Both model folders created
✓ model.json files present
✓ .bin weight files present
✓ scaler_params.json files present
✓ Test accuracy printed in console

## Notes

- Training takes 2-5 minutes per model
- Models are optimized for browser usage
- Weights are quantized for smaller file size
- Models are retrained each time you run the scripts
- You can adjust epochs, batch size, and architecture in the scripts

## Next Steps

After training:
1. Verify model files are created
2. Copy models to user/models/ folder
3. Update HTML files to load trained models
4. Test predictions in browser

## References

- Pima Indians Diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- UCI Heart Disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- TensorFlow.js: https://www.tensorflow.org/js
