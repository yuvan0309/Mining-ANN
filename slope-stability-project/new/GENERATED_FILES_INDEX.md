# Generated Files Index for ML Writeup

## 📂 Directory Structure

```
new/
├── models/
│   ├── COMPREHENSIVE_TRAINING_OUTPUT.xlsx          ⭐ Main Excel workbook (6 sheets)
│   ├── TRAINING_SUMMARY.md                         📝 Markdown summary
│   ├── test_results_ranked.csv                     📊 Ranked test results
│   ├── detailed_metrics_all_models.csv             📊 Detailed metrics with overfitting
│   ├── top_two_models.csv                          📊 Top 2 models comparison
│   ├── test_predictions_gradient_boosting.csv      📈 GB predictions
│   ├── test_predictions_xgboost.csv                📈 XGBoost predictions
│   ├── test_predictions.csv                        📈 Combined predictions
│   ├── training_results.csv                        📊 All models training results
│   ├── test_results.csv                            📊 All models test results
│   ├── comparison_summary.csv                      📊 Phase comparison
│   └── results_summary.json                        🔧 Structured results
│
└── visualizations/
    └── all_models_histograms/
        ├── gradient_boosting_histogram.png         📊 GB error distribution
        ├── gradient_boosting_histogram.pdf         📄 GB error distribution (PDF)
        ├── xgboost_histogram.png                   📊 XGBoost error distribution
        ├── xgboost_histogram.pdf                   📄 XGBoost error distribution (PDF)
        ├── all_models_comparison.png               📊 Side-by-side comparison
        ├── all_models_comparison.pdf               📄 Comparison (PDF)
        ├── error_statistics_all_models.csv         📊 Statistical summary
        └── error_statistics_all_models.xlsx        📄 Statistical summary (Excel)
```

---

## 📊 File Descriptions

### Excel Workbook (Main Output)
**File**: `models/COMPREHENSIVE_TRAINING_OUTPUT.xlsx`

#### Sheet 1: Overview
- Project information
- Dataset statistics (361 total, 288 train, 73 test)
- Best model identification
- Configuration details (StandardScaler, 5-fold CV, random_state=42)

#### Sheet 2: Training Results
- All 6 models' training phase metrics
- Columns: Model, R² Score, RMSE, MAE

#### Sheet 3: Test Results (Ranked)
- All models ranked by Test R²
- Columns: Rank, Model, R² Score, RMSE, MAE

#### Sheet 4: All Phases Comparison
- Combined training and testing data
- Shows phase-wise performance

#### Sheet 5: Detailed Metrics
- **Overfitting analysis** for each model
- Training vs Testing comparison
- Generalization assessment (Excellent/Good/Fair/Poor)
- Columns: Rank, Model, Training metrics, Test metrics, Overfitting Gap %, Generalization

#### Sheet 6: Top Two Models
- Head-to-head comparison of top 2 performers
- Formatted for easy paper inclusion

---

### CSV Files (Individual Exports)

1. **test_results_ranked.csv**
   - Quick reference for rankings
   - Use in paper tables

2. **detailed_metrics_all_models.csv**
   - Complete metrics with overfitting analysis
   - Best for comprehensive tables

3. **top_two_models.csv**
   - Focused comparison
   - Use for results discussion

4. **error_statistics_all_models.csv**
   - Statistical properties of errors
   - Includes: mean, std, min, max, median, skewness, kurtosis, normality test
   - Use for statistical analysis section

---

### Visualizations (Publication Quality - 300 DPI)

#### Individual Model Histograms
Each histogram includes:
- Error distribution bars (observed)
- Bell curve overlay (normal distribution fit)
- Mean error line (red dashed)
- ±1 standard deviation lines (orange dotted)
- Zero error reference line (green solid)
- Statistics box (9 metrics)
- Sample size indicator

**Gradient Boosting**:
- `gradient_boosting_histogram.png` (for presentations)
- `gradient_boosting_histogram.pdf` (for paper submission)

**XGBoost**:
- `xgboost_histogram.png` (for presentations)
- `xgboost_histogram.pdf` (for paper submission)

#### Comparison Plot
- `all_models_comparison.png` - Grid layout with all models
- `all_models_comparison.pdf` - PDF version
- Currently shows 2 models side-by-side
- Consistent formatting across subplots

---

## 🎯 Quick Usage Guide

### For Abstract:
```
"Six machine learning models were trained and evaluated. 
Gradient Boosting achieved the highest test R² score of 0.9426."
```

### For Methods Table:
Use **Sheet 1 (Overview)** from Excel:
- Dataset: 361 samples
- Split: 80/20 (288 train / 73 test)
- Validation: 5-fold cross-validation
- Scaling: StandardScaler
- Random seed: 42

### For Results Table:
Use **Sheet 3 (Test Results Ranked)** or `test_results_ranked.csv`:

| Rank | Model | R² | RMSE | MAE |
|------|-------|-----|------|-----|
| 1 | Gradient Boosting | 0.9426 | 0.0834 | 0.0563 |
| 2 | XGBoost | 0.9420 | 0.0838 | 0.0597 |

### For Discussion Section:
Use **Sheet 5 (Detailed Metrics)**:
- Both top models show ~5.6% overfitting gap
- Generalization: Excellent
- Minimal difference in performance (0.06% R² points)

### For Figures:
1. **Figure 1**: Use `gradient_boosting_histogram.pdf`
   - Caption: "Error distribution of Gradient Boosting model with normal distribution overlay"
   
2. **Figure 2**: Use `xgboost_histogram.pdf`
   - Caption: "Error distribution of XGBoost model with normal distribution overlay"
   
3. **Figure 3**: Use `all_models_comparison.pdf`
   - Caption: "Comparison of error distributions for all models"

---

## 📈 Key Statistics Ready for Paper

### Model Performance (Test Set)

**Gradient Boosting:**
- R² Score: 0.9426
- RMSE: 0.0834
- MAE: 0.0563
- Mean Error: 0.0195 ± 0.0811
- Skewness: 1.43 (right-skewed)
- Kurtosis: 5.65 (leptokurtic)

**XGBoost:**
- R² Score: 0.9420
- RMSE: 0.0838
- MAE: 0.0597
- Mean Error: 0.0112 ± 0.0830
- Skewness: 1.15 (right-skewed)
- Kurtosis: 4.43 (leptokurtic)

---

## ✅ All Files Generated Successfully

- [x] 1 Excel workbook (6 sheets)
- [x] 1 Markdown summary
- [x] 4 CSV exports (metrics & statistics)
- [x] 4 histogram images (2 models × 2 formats)
- [x] 2 comparison plots (PNG + PDF)
- [x] 1 statistical analysis table (CSV + Excel)

**Total: 15 files ready for ML writeup** 📚

---

## 🔄 To Generate More Histograms

If you have prediction files for other models (LightGBM, Random Forest, SVM, ANN), ensure they're in:
```
models/test_predictions_<model_name>.csv
```

Format required:
```csv
Actual FoS,Predicted FoS,Error
1.234,1.245,0.011
...
```

Then re-run:
```bash
python generate_all_models_histograms.py
```

---

**Last Updated**: December 2024  
**Status**: ✅ All primary files generated and ready for writeup
