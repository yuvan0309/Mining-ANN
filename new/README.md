# FoS Prediction System with Ru (Pore Pressure Ratio)

## Overview

This new implementation incorporates **Ru (pore pressure ratio)** values into the Factor of Safety (FoS) prediction model based on **Bishop's Simplified Method** for slope stability analysis.

## Key Features

### 1. **Ru Incorporation**
- Ru (pore pressure ratio) is included as a feature in the model
- Based on the formula: `u = Ru × σv` where:
  - `u` = pore water pressure
  - `Ru` = pore pressure ratio
  - `σv` = total vertical stress

### 2. **Train-Test Split Strategy**
- **80% Training Data**: Used to train ALL models
- **20% Testing Data**: Used ONLY for the best performing model
- Ensures fair comparison and prevents overfitting

### 3. **Model Selection Process**
1. Train all 6 models on 80% training data:
   - SVM (Support Vector Machine)
   - Random Forest
   - XGBoost
   - LightGBM
   - Gradient Boosting
   - ANN (Artificial Neural Network)
2. Evaluate each model on training data (R², RMSE, MAE)
3. Select the best performing model based on R² score
4. Test ONLY the best model on 20% test data

## Directory Structure

```
new/
├── data/
│   └── Overall Data.csv         # Input data with Ru values
├── models/
│   ├── model_*.pkl              # Trained models
│   ├── best_model.pkl           # Best performing model
│   ├── scaler.pkl               # Feature scaler
│   └── results_summary.json     # All results
├── visualizations/
│   ├── training_comparison_all_models.png
│   ├── training_results_table.png
│   ├── *_test_comparison.png    # Best model test results
│   └── *_test_line_graph.png
├── results/                     # Additional outputs
├── data_ingestion.py            # Data loading with Ru
├── train_models.py              # Model training
├── generate_visualizations.py   # Visualization generation
├── main_pipeline.py             # Complete pipeline
└── README.md                    # This file
```

## Usage

### Running the Complete Pipeline

```bash
cd "new"
python main_pipeline.py
```

This will:
1. Load data with Ru values
2. Split into 80-20
3. Train all models on 80%
4. Select best model
5. Test best model on 20%
6. Generate all visualizations
7. Save models and results

### Using Individual Modules

```python
# Load data
from data_ingestion import load_and_prepare_data, train_test_split_data
X, y, df = load_and_prepare_data("data/Overall Data.csv", include_ru=True)
X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)

# Train models
from train_models import FoSModelTrainer
trainer = FoSModelTrainer(X_train, y_train, X_test, y_test)
trainer.train_all_models()
trainer.test_best_model()

# Generate visualizations
from generate_visualizations import create_all_visualizations
create_all_visualizations("models/results_summary.json", "visualizations")
```

## Visualizations Generated

### Training Phase (All Models, 80% Data)
1. **training_comparison_all_models.png**: Bar charts comparing R², RMSE, MAE
2. **training_results_table.png**: Table with all training metrics

### Testing Phase (Best Model Only, 20% Data)
1. **{model}_test_comparison.png**: Actual vs Predicted scatter plot
2. **{model}_test_line_graph.png**: Line graph with actual and predicted values
3. **{model}_train_vs_test.png**: Training vs Testing performance comparison

## Web Integration

### Making Ru Optional in Website

The prediction function can accept Ru as an optional parameter:

```python
def predict_fos(cohesion, friction_angle, unit_weight, ru=None):
    """
    Predict FoS with optional Ru value.
    
    Parameters:
    -----------
    cohesion : float
        Cohesion (kPa)
    friction_angle : float
        Friction angle (degrees)
    unit_weight : float
        Unit weight (kN/m³)
    ru : float, optional
        Pore pressure ratio (0-1). If None, defaults to 0.
    """
    import joblib
    import numpy as np
    
    # Load model and scaler
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Prepare input
    ru_value = 0.0 if ru is None else ru
    X = np.array([[cohesion, friction_angle, unit_weight, ru_value]])
    X_scaled = scaler.transform(X)
    
    # Predict
    fos_prediction = model.predict(X_scaled)[0]
    
    return fos_prediction
```

### Website Form Example

```html
<form id="fosForm">
    <label>Cohesion (kPa):</label>
    <input type="number" name="cohesion" required>
    
    <label>Friction Angle (°):</label>
    <input type="number" name="friction_angle" required>
    
    <label>Unit Weight (kN/m³):</label>
    <input type="number" name="unit_weight" required>
    
    <label>Ru (Pore Pressure Ratio) - Optional:</label>
    <input type="number" name="ru" min="0" max="1" step="0.01" placeholder="0.0">
    <small>Leave blank or 0 for dry conditions. Range: 0-1</small>
    
    <button type="submit">Predict FoS</button>
</form>
```

## Technical Details

### Bishop's Simplified Method with Ru

The Factor of Safety is calculated considering pore pressure:

```
F = Σ[c'l + (W - RuW)tanφ'] / [Σ W sinα × (1 + (tanα tanφ'/F))]
```

Where:
- `c'` = effective cohesion
- `φ'` = effective friction angle
- `W` = slice weight
- `Ru` = pore pressure ratio
- `α` = base angle

### Model Training Details

- **Feature Scaling**: StandardScaler for normalization
- **Random State**: 42 for reproducibility
- **Validation**: Training metrics calculated on 80% data
- **Testing**: Only best model tested on unseen 20% data

### Performance Metrics

- **R²** (Coefficient of Determination): Model accuracy (0-1, higher better)
- **RMSE** (Root Mean Square Error): Prediction error (lower better)
- **MAE** (Mean Absolute Error): Average absolute error (lower better)

## Important Notes

⚠️ **Testing Protocol**:
- ALL models are trained on 80% data
- ONLY the best performing model is tested on 20% data
- This prevents data leakage and ensures fair model selection

📊 **Ru Values**:
- Typical ranges:
  - Dry slope: 0.0
  - Damp soil: 0.1 - 0.2
  - Saturated slope: 0.3 - 0.5
  - Fully submerged: 0.5 - 1.0

## References

- Bishop's Simplified Method (1955)
- Rocscience Slide2 documentation
- instructions.txt (attached in project)
