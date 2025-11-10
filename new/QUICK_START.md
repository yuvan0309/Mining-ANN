# Quick Start Guide - FoS Prediction with Ru

## 🚀 Running the Pipeline

```bash
cd "/home/inanotherlife/Mining ANN/new"
source ../venv/bin/activate
python main_pipeline.py
```

## 📊 What Gets Generated

### Models (in `models/` directory):
- `best_model.pkl` - Use this for predictions! (Gradient Boosting)
- `scaler.pkl` - Required for input preprocessing
- `results_summary.json` - All metrics and results

### Visualizations (in `visualizations/` directory):
- `training_comparison_all_models.png` - Compare all 6 models
- `training_results_table.png` - Metrics table
- `gradient_boosting_test_comparison.png` - Test scatter plot
- `gradient_boosting_test_line_graph.png` - Test line graph
- `gradient_boosting_train_vs_test.png` - Performance comparison

## 💻 Making a Prediction

```python
import joblib
import numpy as np

# Load models
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Input values
cohesion = 35.0          # kPa
friction_angle = 28.0     # degrees
unit_weight = 19.5       # kN/m³
ru = 0.3                 # 0-1 (optional, defaults to 0)

# Prepare and predict
X = np.array([[cohesion, friction_angle, unit_weight, ru]])
X_scaled = scaler.transform(X)
fos = model.predict(X_scaled)[0]

print(f"Predicted FoS: {fos:.3f}")
```

## 🌐 Website Integration (Minimal Example)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    X = np.array([[
        data['cohesion'],
        data['friction_angle'],
        data['unit_weight'],
        data.get('ru', 0.0)  # Optional, defaults to 0
    ]])
    
    X_scaled = scaler.transform(X)
    fos = model.predict(X_scaled)[0]
    
    return jsonify({'fos': float(fos)})

app.run()
```

## 📝 Ru (Pore Pressure Ratio) Guide

| Ru Value | Condition | Description |
|----------|-----------|-------------|
| 0.0 | Dry | No pore pressure (default) |
| 0.1-0.2 | Damp | Partially saturated soil |
| 0.3-0.5 | Saturated | Fully saturated slope |
| 0.5-1.0 | Submerged | Underwater conditions |

## 🎯 Model Performance

- **Best Model**: Gradient Boosting
- **Training R²**: 0.9982 (99.82% accuracy)
- **Testing R²**: 0.9060 (90.6% accuracy)
- **Testing RMSE**: 0.1067
- **Testing MAE**: 0.0654

## ✅ Key Points

1. ✅ **80-20 Split**: 80% training, 20% testing
2. ✅ **All Models Trained**: 6 models compared on training data
3. ✅ **Best Model Selected**: Gradient Boosting chosen based on R²
4. ✅ **Only Best Tested**: Only Gradient Boosting tested on 20% test data
5. ✅ **Ru is Optional**: Can be omitted (defaults to 0.0)

## 📁 File Structure

```
new/
├── main_pipeline.py          # Run this!
├── data_ingestion.py          # Data loading with Ru
├── train_models.py            # Model training
├── generate_visualizations.py # Create plots
├── models/
│   ├── best_model.pkl        # Use this for predictions
│   └── scaler.pkl             # Use this too
├── visualizations/            # All generated plots
├── data/
│   └── Overall Data.csv       # Input data
├── README.md                  # Full documentation
└── IMPLEMENTATION_SUMMARY.md  # Detailed summary
```

## 🆘 Troubleshooting

### Import Error
```bash
source ../venv/bin/activate
pip install scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn
```

### Model Not Found
```bash
# Run the pipeline first
python main_pipeline.py
```

### Prediction Error
```python
# Ensure Ru is between 0 and 1
ru = max(0.0, min(1.0, ru))
```

## 📚 Documentation

- **Full README**: `README.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Bishop's Method**: `../instructions.txt`
- **Results JSON**: `models/results_summary.json`

## 🎓 Understanding Results

### FoS Interpretation:
- **FoS ≥ 1.5**: ✅ Safe (stable slope)
- **1.0 ≤ FoS < 1.5**: ⚠️ Marginal (monitor)
- **FoS < 1.0**: ❌ Unsafe (failure risk)

### Model Metrics:
- **R²**: How well model fits data (0-1, higher better)
- **RMSE**: Root Mean Square Error (lower better)
- **MAE**: Mean Absolute Error (lower better)

---

**Need Help?** Check the full documentation in `README.md` or `IMPLEMENTATION_SUMMARY.md`
