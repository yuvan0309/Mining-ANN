# Output Files Guide - CSV and Excel Results

## 📊 Overview

All results are now available in **CSV** and **Excel** formats as requested. The 20% testing is performed **only on the best performing model** (Gradient Boosting).

---

## 📁 File Locations

### CSV Files (in `models/` directory):

1. **`training_results.csv`** - All 6 models trained on 80% data
2. **`test_results.csv`** - Best model only (Gradient Boosting) on 20% test data
3. **`test_predictions.csv`** - Detailed predictions for 73 test samples

### Excel File (in `visualizations/` directory):

- **`all_results.xlsx`** - Comprehensive Excel workbook with 3 sheets

---

## 📋 File Details

### 1. Training Results (`training_results.csv`)

**Purpose**: Compare all 6 models trained on 80% of the data

**Columns**:
- `Model` - Model name
- `R² Score` - Coefficient of determination (higher is better)
- `RMSE` - Root Mean Square Error (lower is better)
- `MAE` - Mean Absolute Error (lower is better)

**Content**:
```
Model,R² Score,RMSE,MAE
Gradient Boosting,0.9982,0.0151,0.0023
XGBoost,0.9982,0.0151,0.0032
Random Forest,0.9924,0.0313,0.0220
LightGBM,0.9872,0.0407,0.0297
SVM,0.9570,0.0746,0.0616
ANN,0.9316,0.0940,0.0694
```

**Total Rows**: 6 (one per model)

---

### 2. Test Results (`test_results.csv`)

**Purpose**: Performance of best model (Gradient Boosting) on 20% test data

**Columns**:
- `Model` - Best model name
- `R² Score` - Test R² score
- `RMSE` - Test Root Mean Square Error
- `MAE` - Test Mean Absolute Error

**Content**:
```
Model,R² Score,RMSE,MAE
Gradient Boosting,0.9060,0.1067,0.0654
```

**Total Rows**: 1 (best model only)

**⚠️ Important**: Only the best performing model was tested on the 20% test data to prevent data leakage.

---

### 3. Test Predictions (`test_predictions.csv`)

**Purpose**: Detailed predictions for each test sample

**Columns**:
- `Actual FoS` - True Factor of Safety value
- `Predicted FoS` - Model's predicted value
- `Error` - Difference (Actual - Predicted)

**Sample Data**:
```
Actual FoS,Predicted FoS,Error
0.683,0.6914,-0.0084
1.520,1.5373,-0.0173
0.688,0.7356,-0.0476
1.568,1.5835,-0.0155
...
```

**Total Rows**: 73 (20% of 361 samples)

---

### 4. Excel Workbook (`all_results.xlsx`)

**Purpose**: All results in one formatted Excel file

**Sheets**:

1. **"Training Results (All Models)"** - Same as `training_results.csv`
   - 6 models, 80% training data
   - Sorted by R² Score (best to worst)

2. **"Test Results (Best Model)"** - Same as `test_results.csv`
   - Gradient Boosting only
   - 20% test data

3. **"Test Predictions"** - Same as `test_predictions.csv`
   - 73 test samples with predictions
   - Actual vs Predicted comparison

**Formatting**:
- ✅ Professional header styling (blue background, white text)
- ✅ Borders on all cells
- ✅ Numbers formatted to 4 decimal places
- ✅ Auto-adjusted column widths
- ✅ Centered alignment

---

## 📊 Key Results Summary

### Training Phase (80% data - 288 samples)

| Rank | Model | R² Score | RMSE | MAE |
|------|-------|----------|------|-----|
| 🏆 1 | Gradient Boosting | 0.9982 | 0.0151 | 0.0023 |
| 2 | XGBoost | 0.9982 | 0.0151 | 0.0032 |
| 3 | Random Forest | 0.9924 | 0.0313 | 0.0220 |
| 4 | LightGBM | 0.9872 | 0.0407 | 0.0297 |
| 5 | SVM | 0.9570 | 0.0746 | 0.0616 |
| 6 | ANN | 0.9316 | 0.0940 | 0.0694 |

### Testing Phase (20% data - 73 samples)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Gradient Boosting | **0.9060** | **0.1067** | **0.0654** |

---

## 🎯 Understanding the Metrics

### R² Score (Coefficient of Determination)
- **Range**: 0 to 1
- **Meaning**: Proportion of variance in FoS explained by the model
- **Interpretation**:
  - 0.90 = 90% of variance explained (Excellent)
  - 0.95 = 95% of variance explained (Outstanding)
  - 0.99 = 99% of variance explained (Near-perfect)

### RMSE (Root Mean Square Error)
- **Units**: Same as FoS
- **Meaning**: Average magnitude of prediction errors
- **Interpretation**:
  - Lower is better
  - 0.1067 means average error of ~0.11 FoS units

### MAE (Mean Absolute Error)
- **Units**: Same as FoS
- **Meaning**: Average absolute difference between actual and predicted
- **Interpretation**:
  - Lower is better
  - 0.0654 means average error of ~0.07 FoS units

---

## 💻 Using the CSV Files

### Python Example:
```python
import pandas as pd

# Load training results
training_df = pd.read_csv('models/training_results.csv')
print(training_df)

# Load test results
test_df = pd.read_csv('models/test_results.csv')
print(f"Best Model: {test_df['Model'].iloc[0]}")
print(f"Test R²: {test_df['R² Score'].iloc[0]:.4f}")

# Load predictions
predictions_df = pd.read_csv('models/test_predictions.csv')
print(f"Average Error: {predictions_df['Error'].abs().mean():.4f}")
```

### Excel Example:
1. Open `visualizations/all_results.xlsx`
2. Navigate between sheets using tabs at bottom
3. All data is formatted and ready for analysis
4. Can be easily copied to reports or presentations

---

## 📈 Why Only Best Model for Testing?

**Training Phase (80%)**:
- ✅ All 6 models trained
- ✅ All 6 models evaluated
- ✅ Best model selected (Gradient Boosting)

**Testing Phase (20%)**:
- ✅ Only Gradient Boosting tested
- ❌ Other 5 models NOT tested

**Reason**: 
- Prevents **data leakage**
- Avoids **overfitting** to test set
- Ensures **unbiased evaluation**
- Standard **machine learning best practice**

---

## 📁 Complete File Structure

```
new/
├── models/
│   ├── training_results.csv       ← All 6 models (80% data)
│   ├── test_results.csv           ← Best model only (20% data)
│   ├── test_predictions.csv       ← 73 predictions
│   ├── results_summary.json       ← JSON format (backup)
│   ├── best_model.pkl             ← Trained model
│   ├── scaler.pkl                 ← Feature scaler
│   └── model_*.pkl                ← All 6 models
│
└── visualizations/
    ├── all_results.xlsx           ← Excel workbook (3 sheets)
    ├── training_results.csv       ← Copy for convenience
    ├── test_results.csv           ← Copy for convenience
    ├── test_predictions.csv       ← Copy for convenience
    └── *.png                      ← Visualization images
```

---

## ✅ Verification Checklist

- [x] Training results CSV created (6 models)
- [x] Test results CSV created (1 model)
- [x] Test predictions CSV created (73 samples)
- [x] Excel file created with 3 sheets
- [x] All files properly formatted
- [x] Numbers accurate to 4 decimal places
- [x] Only best model tested on 20% data
- [x] CSV files copied to visualizations/

---

## 🆘 Quick Reference

**To view training comparison**:
```bash
cat models/training_results.csv
```

**To view test results**:
```bash
cat models/test_results.csv
```

**To open Excel file**:
```bash
libreoffice visualizations/all_results.xlsx
# or
open visualizations/all_results.xlsx  # macOS
```

**To count test samples**:
```bash
wc -l models/test_predictions.csv
# Output: 74 (73 data rows + 1 header)
```

---

## 📚 Related Documentation

- **QUICK_START.md** - How to run pipeline
- **README.md** - Complete documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **formula file with steps.md** - Bishop's Method

---

**✨ All output files are ready for analysis, reporting, and web integration!**
