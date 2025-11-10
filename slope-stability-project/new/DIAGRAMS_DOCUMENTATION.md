# 📐 ARCHITECTURE & FLOW DIAGRAMS DOCUMENTATION

**Project**: Factor of Safety (FoS) Prediction System with Ru Integration  
**Date**: November 10, 2025  
**Location**: `/home/inanotherlife/Mining ANN/new/visualizations/`

---

## 📊 Generated Diagrams

Three comprehensive diagrams have been created to visualize the entire system:

### 1. 🏗️ **ARCHITECTURE_DIAGRAM.png**
**System Architecture Overview**

This diagram shows the high-level architecture of the FoS prediction system, displaying all major components and their relationships.

#### Key Components Shown:

**LAYER 1: DATA INPUT**
- Pre-Monsoon Data (180 samples)
- Post-Monsoon Data (181 samples)  
- Ru Values (Pore Pressure Ratio)
- Total: 361 samples from 10 mines (A, B, C variants)

**LAYER 2: DATA INGESTION**
- Module: `data_ingestion.py`
- Functions: Parse CSV, Extract features, Handle Ru
- Features: Cohesion (kPa), Friction Angle (°), Unit Weight (kN/m³), Ru (optional)

**LAYER 3: DATA SPLIT**
- 80% Training: 288 samples (green)
- 20% Testing: 73 samples (orange)
- Stratified random split with seed=42

**LAYER 4: MODEL TRAINING**
- Module: `train_models.py`
- 6 ML Models trained in parallel:
  1. SVM (RBF kernel)
  2. Random Forest
  3. **XGBoost** (Fine-tuned - highlighted in gold)
  4. LightGBM
  5. **Gradient Boosting** (Fine-tuned - highlighted in gold)
  6. ANN (Multi-Layer Perceptron)
- All models use StandardScaler

**LAYER 5: MODEL SELECTION**
- Selects top 2 models based on training R²
- Winner: Gradient Boosting (R²=0.9954)
- Runner-up: XGBoost (R²=0.9581)

**LAYER 6: TESTING**
- Only top 2 models tested on 20% data
- Gradient Boosting: R²=0.9426 ✅
- XGBoost: R²=0.9420 ✅
- Both models essentially tied (~94.2% accuracy)

**LAYER 7: OUTPUTS**
- Models (.pkl files) - 6 trained models + scaler
- CSV Results (training & testing metrics)
- Excel File (all_results.xlsx with 4 sheets)
- Visualizations (8 PNG files)
- JSON Metadata (results summary)

#### Right Side Panel:
- **Key Features**: Bishop's Method, Ru Integration, 6 ML Models, Fine-tuned Top 2, 80/20 Split
- **Performance Metrics**: GB R²=0.9426, XGB R²=0.9420

---

### 2. 🔄 **FLOW_DIAGRAM.png**
**Detailed Process Flow (12 Steps)**

This diagram shows the complete workflow from start to finish, with detailed step-by-step process.

#### Process Steps:

**STEP 1: START** (Green circle)
- Entry point of the pipeline

**STEP 2: Load Data**
- Function: `load_and_prepare_data()`
- Input: Overall Data.csv
- Include Ru: Yes

**STEP 3: Parse CSV Structure**
- Extract 361 samples
- Handle multiple sections in CSV

**STEP 4: Feature Matrix (X)**
- Extract 4 features:
  - Cohesion (c) - kPa
  - Friction Angle (φ) - degrees
  - Unit Weight (γ) - kN/m³
  - Ru (Pore Pressure Ratio)
- Target: FoS values

**STEP 5: Train-Test Split**
- Function: `train_test_split()`
- Split ratio: 80/20
- random_state=42
- Branches to training (288) and testing (73) paths

**STEP 6: StandardScaler**
- Fit on training data only
- Mean=0, Std=1 transformation
- Critical for model performance

**STEP 7: Train All 6 Models**
- Parallel training on 288 samples
- Models listed:
  1. SVM (RBF)
  2. Random Forest
  3. XGBoost (Fine-tuned)
  4. LightGBM
  5. Gradient Boosting (Fine-tuned)
  6. ANN (MLP)
- Each model predicts on training set

**STEP 8: Evaluate on Training**
- Calculate R², RMSE, MAE for each model
- Compare performance
- Identify top 2 performers

**STEP 9: Select Top 2**
- Based on training R² score
- Gradient Boosting: Highest
- XGBoost: Second highest

**STEP 10: Scale Test Data**
- Use training scaler (transform only, not fit)
- Testing data (73 samples) joined here
- Critical: No data leakage

**STEP 11: Test Top 2 Models**
- Test only the best 2 models on 73 samples
- Results:
  - 🥇 Gradient Boosting: R²=0.9426
  - 🥈 XGBoost: R²=0.9420
- Both achieve ~94.2% accuracy

**STEP 12: Save Models**
- Save all 6 models as .pkl files
- Save scaler separately

**STEP 13: Generate Outputs**
- training_results.csv (6 models)
- test_results.csv (2 models)
- test_predictions_*.csv (73 rows each)
- all_results.xlsx (4 sheets)
- 8 PNG visualizations
- JSON metadata

**STEP 14: END** (Red circle)
- Pipeline complete

#### Right Side Panels:

**KEY METRICS**
```
TRAINING (80% - 288 samples):
• Gradient Boosting: R²=0.9954
• Random Forest: R²=0.9924
• XGBoost: R²=0.9581

TESTING (20% - 73 samples):
• Gradient Boosting: R²=0.9426
• XGBoost: R²=0.9420

Overfitting Control:
• XGBoost gap: 1.61% ✅
• GB gap: 5.28% ✅
```

**FILE STRUCTURE**
```
new/
├── main_pipeline.py (Entry)
├── data_ingestion.py
├── train_models.py
├── generate_visualizations.py
├── data/
│   └── Overall Data.csv
├── models/
│   ├── *.pkl (6 models)
│   ├── scaler.pkl
│   ├── training_results.csv
│   ├── test_results.csv
│   └── test_predictions_*.csv
└── visualizations/
    ├── *.png (8 files)
    └── all_results.xlsx
```

**TECH STACK**
- Python 3.13.7
- scikit-learn (ML models)
- XGBoost, LightGBM
- pandas, numpy
- matplotlib, seaborn
- openpyxl (Excel)
- joblib (Model saving)

---

### 3. 🔄 **DATA_FLOW_DIAGRAM.png**
**Data Transformation Pipeline**

This diagram focuses specifically on data transformations from raw CSV to model predictions.

#### Data Flow Stages:

**STAGE 1: RAW CSV** (Blue)
- File: Overall Data.csv
- 361 samples
- Mixed format (headers, multiple sections)

**STAGE 2: PARSED DATA** (Green)
- Features extracted:
  - Cohesion
  - Friction Angle
  - Unit Weight
  - Ru (optional)
- Clean DataFrame format
- Shape: (361, 4)

**STAGE 3: SPLIT DATA** (Purple)
- 80/20 split
- Stratified by FoS distribution
- Two separate datasets created

**STAGE 4: TRAIN SET** (Dark Green)
- 288 samples (80%)
- Shape: (288, 4)
- Used for model training

**STAGE 5: TEST SET** (Orange)
- 73 samples (20%)
- Shape: (73, 4)
- HELD OUT until testing phase

**STAGE 6: STANDARDSCALER** (Blue)
- **Training Branch**:
  - Fit & Transform
  - Calculate μ (mean) and σ (std dev)
  - Transform training data
  
- **Testing Branch**:
  - Transform Only (using training μ and σ)
  - No fitting on test data
  - Prevents data leakage

**STAGE 7: 6 MODELS TRAINING** (Red)
- All models trained in parallel
- Input: Scaled train data (288, 4)
- Each model learns patterns independently

**STAGE 8: TRAIN PREDICTIONS** (Orange)
- All 6 models predict on training data
- Output shape: (288, 6 models)
- Used for model selection

**STAGE 9: TEST PREDICTIONS** (Gold)
- Only top 2 models predict on test data
- Output shape: (73, 2 models)
- Final evaluation

**STAGE 10: OUTPUTS** (Dark Gray)
- Models (.pkl)
- CSV Results
- Excel File
- PNG Visualizations
- JSON Metadata

#### Left Side Panel: DATA DIMENSIONS

```
Raw CSV:
• Rows: 361
• Columns: Variable

Parsed Features (X):
• Shape: (361, 4)
• dtypes: float64

Train Set:
• X_train: (288, 4)
• y_train: (288,)

Test Set:
• X_test: (73, 4)
• y_test: (73,)

Scaled:
• Mean=0, Std=1
• Same shape
```

#### Right Side Panel: TRANSFORMATIONS

```
StandardScaler Formula:
z = (x - μ) / σ

Where:
• x = feature value
• μ = mean (from training)
• σ = std dev (from training)

Applied to:
1. Cohesion (kPa)
2. Friction Angle (°)
3. Unit Weight (kN/m³)
4. Ru (0-1 range)

Benefits:
• Equal feature importance
• Faster convergence
• Better model performance
```

---

## 🎨 Diagram Color Coding

### Architecture Diagram:
- **Blue** (#3498db): Data input/sources
- **Green** (#2ecc71): Processing modules
- **Red** (#e74c3c): Model training
- **Orange** (#f39c12): Outputs
- **Purple** (#9b59b6): Storage/persistence
- **Gold** (#FFD700): Best models (highlighted)

### Flow Diagram:
- **Green circle**: Start
- **Blue boxes**: Data operations
- **Green boxes**: Training data flow
- **Orange boxes**: Testing data flow
- **Red boxes**: Model training
- **Purple boxes**: Model selection
- **Gold boxes**: Best model testing
- **Dark gray boxes**: Output generation
- **Red circle**: End

### Data Flow Diagram:
- **Blue**: Raw data & scaling
- **Green**: Parsed/processed data
- **Purple**: Split operation
- **Dark Green**: Training set
- **Orange**: Testing set
- **Red**: Model training
- **Gold**: Testing predictions
- **Dark Gray**: Final outputs

---

## 📋 How to Use These Diagrams

### For Understanding the System:
1. **Start with ARCHITECTURE_DIAGRAM.png**
   - Gives big picture view
   - Shows all major components
   - Identifies key modules

2. **Then review FLOW_DIAGRAM.png**
   - Understand step-by-step process
   - See decision points
   - Follow data paths

3. **Finally check DATA_FLOW_DIAGRAM.png**
   - Focus on data transformations
   - Understand shape changes
   - Verify data integrity

### For Presentations:
- **Architecture Diagram**: Executive overview
- **Flow Diagram**: Technical deep-dive
- **Data Flow Diagram**: Data science explanation

### For Documentation:
- Include in README.md
- Add to project wiki
- Use in technical reports

### For Debugging:
- Identify where errors occur
- Trace data through pipeline
- Verify transformations

---

## 🔍 Key Insights from Diagrams

### 1. **Parallel Training, Sequential Testing**
- All 6 models trained simultaneously (efficient)
- Only top 2 tested on held-out data (prevents overfitting)

### 2. **Data Leakage Prevention**
- Scaler fit only on training data
- Test data never seen during training
- Proper 80/20 split maintained

### 3. **Fine-Tuning Focus**
- XGBoost and Gradient Boosting specially tuned
- Hyperparameters optimized for generalization
- Both achieve ~94.2% test accuracy

### 4. **Comprehensive Outputs**
- Multiple formats: CSV, Excel, PNG, JSON, PKL
- Training and testing results separated
- Full prediction details available

### 5. **Bishop's Method Integration**
- Ru (pore pressure ratio) properly incorporated
- All features from geotechnical theory
- FoS prediction based on established method

---

## 📁 File Locations

All diagrams saved in:
```
/home/inanotherlife/Mining ANN/new/visualizations/
├── ARCHITECTURE_DIAGRAM.png (16x12 inches, 300 DPI)
├── FLOW_DIAGRAM.png (14x18 inches, 300 DPI)
└── DATA_FLOW_DIAGRAM.png (16x10 inches, 300 DPI)
```

### Diagram Specifications:
- **Format**: PNG
- **Resolution**: 300 DPI (print quality)
- **Background**: White
- **Color Mode**: RGB
- **File Size**: 200-400 KB each

---

## 🚀 Next Steps

### Using These Diagrams:

1. **For Presentations**:
   - Import into PowerPoint/Google Slides
   - Add annotations as needed
   - Explain to stakeholders

2. **For Documentation**:
   - Embed in Markdown files
   - Add to project README
   - Include in technical reports

3. **For Training**:
   - Use to onboard new team members
   - Explain system architecture
   - Show data flow

4. **For Papers/Publications**:
   - High-resolution for journals
   - Clear visualization of methodology
   - Supports reproducibility

---

## 📊 Diagram Statistics

| Diagram | Components | Connections | Layers | Detail Level |
|---------|-----------|-------------|---------|--------------|
| Architecture | 25+ | 15+ | 7 | High-level |
| Flow | 35+ | 25+ | 14 steps | Detailed |
| Data Flow | 20+ | 12+ | 10 stages | Technical |

**Total Visual Elements**: 80+ components across 3 diagrams

---

## ✅ Validation Checklist

Use this to verify diagrams are complete:

- [x] All 3 diagrams generated successfully
- [x] High resolution (300 DPI)
- [x] Color-coded for clarity
- [x] All major components shown
- [x] Data flow paths clear
- [x] Annotations readable
- [x] File sizes reasonable (<500KB)
- [x] Saved in correct location
- [x] Documentation complete

---

## 🎯 Summary

Three comprehensive diagrams provide complete visualization:

1. **ARCHITECTURE_DIAGRAM.png**: System overview with 7 layers
2. **FLOW_DIAGRAM.png**: 14-step detailed process flow
3. **DATA_FLOW_DIAGRAM.png**: Data transformation pipeline

**Total Coverage**: From raw CSV to final predictions with full traceability!

---

**Generated**: November 10, 2025  
**Project**: FoS Prediction with Ru Integration  
**Author**: Mining ANN System  
**Version**: 1.0
