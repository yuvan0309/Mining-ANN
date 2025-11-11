# Complete Model Training Results

## Training Results (All 6 Models)

| Rank | Model | Training R² | Training RMSE | Training MAE |
|------|-------|-------------|---------------|--------------|
| 1 | **Gradient Boosting** | 0.9954 | 0.0245 | 0.0156 |
| 2 | **Random Forest** | 0.9924 | 0.0313 | 0.0220 |
| 3 | **LightGBM** | 0.9872 | 0.0407 | 0.0297 |
| 4 | **XGBoost** | 0.9581 | 0.0735 | 0.0558 |
| 5 | **SVM** | 0.9570 | 0.0746 | 0.0616 |
| 6 | **ANN** | 0.9316 | 0.0940 | 0.0694 |

---

## Test Results with Overfitting Analysis

| Rank | Model | Test R² | Test RMSE | Test MAE | Overfitting Gap | Generalization |
|------|-------|---------|-----------|----------|-----------------|----------------|
| 1 | **Gradient Boosting** | 0.9426 | 0.0834 | 0.0563 | 5.31% | Good |
| 2 | **XGBoost** | 0.9420 | 0.0838 | 0.0597 | 1.68% | Excellent |

---

**Dataset**: 361 samples (288 training / 73 testing)  Excluding Outluiers  
**Features**: 4 (Cohesion, Friction Angle, Unit Weight, Ru)
**Split Ratio**: 80% / 20%  
**Validation**: 5-fold cross-validation  
**Scaling**: StandardScaler  
**Random State**: 42
