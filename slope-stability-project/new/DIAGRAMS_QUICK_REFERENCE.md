# 🎨 ARCHITECTURE & FLOW DIAGRAMS - QUICK REFERENCE

---

## 📊 3 Diagrams Generated

### 1. 🏗️ ARCHITECTURE_DIAGRAM.png (475 KB)
**System Overview - 7 Layers**

```
┌─────────────────────────────────────────────────┐
│  INPUT: Pre-Monsoon + Post-Monsoon + Ru Data   │
│          (361 samples, 10 mines)                │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  DATA INGESTION (data_ingestion.py)            │
│  Parse CSV → Extract 4 Features                │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  TRAIN-TEST SPLIT (80/20)                      │
│  288 Train | 73 Test                           │
└──────┬────────────────────────┬─────────────────┘
       │                        │
       ▼                        ▼
┌─────────────┐          ┌─────────────┐
│ 80% TRAIN   │          │ 20% TEST    │
│ 288 samples │          │ 73 samples  │
│             │          │ (HELD OUT)  │
└──────┬──────┘          └──────┬──────┘
       │                        │
       ▼                        │
┌──────────────────────────┐   │
│  TRAIN 6 MODELS          │   │
│  • SVM                   │   │
│  • Random Forest         │   │
│  • XGBoost ⭐            │   │
│  • LightGBM              │   │
│  • Gradient Boosting ⭐  │   │
│  • ANN                   │   │
└──────┬───────────────────┘   │
       │                       │
       ▼                       │
┌──────────────────────────┐  │
│  SELECT TOP 2 MODELS     │  │
│  (Best Training R²)      │  │
└──────┬───────────────────┘  │
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  TEST TOP 2 MODELS   │
       │  GB:  R²=0.9426 🥇   │
       │  XGB: R²=0.9420 🥈   │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │      OUTPUTS         │
       │  • Models (.pkl)     │
       │  • CSV Results       │
       │  • Excel (4 sheets)  │
       │  • 8 Visualizations  │
       │  • JSON Metadata     │
       └──────────────────────┘
```

---

### 2. 🔄 FLOW_DIAGRAM.png (713 KB)
**Detailed Process Flow - 14 Steps**

```
START
  ↓
[1] Load Data (data_ingestion.py)
  ↓
[2] Parse CSV Structure
  ↓
[3] Extract Features (c, φ, γ, Ru)
  ↓
[4] Train-Test Split (80/20)
  ↓
  ├─────────────┬─────────────┐
  ↓             ↓             ↓
TRAIN (288)   TEST (73)   (HOLD)
  ↓                         ↓
[5] Fit StandardScaler      │
  ↓                         │
[6] Train 6 Models          │
  ↓                         │
[7] Evaluate Training       │
  ↓                         │
[8] Select Top 2            │
  ↓                         │
[9] Scale Test Data ←───────┘
  ↓
[10] Test Top 2 Models
   • GB:  R²=0.9426
   • XGB: R²=0.9420
  ↓
[11] Save Models (.pkl)
  ↓
[12] Generate Outputs
   • CSV files (5)
   • Excel file (4 sheets)
   • PNG files (8)
   • JSON metadata
  ↓
END
```

**Right Panel Info**:
- Training metrics (all 6 models)
- Testing metrics (top 2 models)
- Overfitting gaps (1.61% & 5.28%)
- File structure tree
- Tech stack list

---

### 3. 🔄 DATA_FLOW_DIAGRAM.png (469 KB)
**Data Transformation Pipeline - 10 Stages**

```
RAW CSV (361 samples)
    ↓
PARSED DATA (361, 4)
[Features: c, φ, γ, Ru]
    ↓
SPLIT DATA (80/20)
    ↓
    ├───────────────┬───────────────┐
    ↓               ↓               ↓
TRAIN (288, 4)  TEST (73, 4)    (HOLD)
    ↓                             ↓
FIT SCALER                        │
    ↓                             ↓
TRANSFORM                   TRANSFORM
(Mean=0, Std=1)            (Using Train μ, σ)
    ↓                             ↓
TRAIN 6 MODELS                    │
(288, 4) → (288, 6)              │
    ↓                             │
TRAIN PREDICTIONS                 │
    ↓                             │
SELECT TOP 2 ─────────────────────┘
    ↓
TEST PREDICTIONS
(73, 4) → (73, 2)
    ↓
OUTPUTS
```

**Left Panel**: Data dimensions at each stage  
**Right Panel**: StandardScaler formula and benefits

---

## 🎯 Key Metrics Shown

### Training Performance (288 samples):
| Model | R² | Status |
|-------|-----|--------|
| Gradient Boosting | 0.9954 | ⭐ Best |
| Random Forest | 0.9924 | |
| LightGBM | 0.9872 | |
| XGBoost | 0.9581 | ⭐ Fine-tuned |
| SVM | 0.9570 | |
| ANN | 0.9316 | |

### Testing Performance (73 samples):
| Model | R² | RMSE | MAE | Winner |
|-------|-----|------|-----|--------|
| **Gradient Boosting** | 0.9426 | 0.0834 | 0.0563 | 🥇 |
| **XGBoost** | 0.9420 | 0.0838 | 0.0597 | 🥈 |

**Difference**: Only 0.06% - essentially tied!

---

## 📐 Diagram Features

### Visual Elements:
- **Color-coded layers** for clarity
- **Arrows show data flow** direction
- **Boxes highlight** key components
- **Annotations explain** each step
- **Metrics displayed** where relevant

### Professional Quality:
- **300 DPI resolution** (print quality)
- **Large format**: 14-18 inches
- **Clean design**: White background
- **Font sizes**: 6-18pt for readability

---

## 📁 File Locations

```bash
/home/inanotherlife/Mining ANN/new/visualizations/
├── ARCHITECTURE_DIAGRAM.png    # 475 KB - System overview
├── FLOW_DIAGRAM.png            # 713 KB - Process flow
└── DATA_FLOW_DIAGRAM.png       # 469 KB - Data pipeline
```

---

## 🚀 Usage Guide

### For Quick Understanding:
1. **Look at ARCHITECTURE_DIAGRAM** first (2 minutes)
   - See the big picture
   - Identify main components

2. **Review FLOW_DIAGRAM** second (5 minutes)
   - Understand detailed steps
   - Follow the process

3. **Check DATA_FLOW_DIAGRAM** last (3 minutes)
   - Focus on data transformations
   - Verify shapes and dimensions

### For Presentations:
- **Slide 1**: Show ARCHITECTURE_DIAGRAM (overview)
- **Slide 2**: Show FLOW_DIAGRAM (details)
- **Slide 3**: Show DATA_FLOW_DIAGRAM (technical)
- **Slide 4**: Show results (from Excel/CSV)

### For Documentation:
```markdown
## System Architecture
![Architecture](visualizations/ARCHITECTURE_DIAGRAM.png)

## Process Flow
![Flow](visualizations/FLOW_DIAGRAM.png)

## Data Pipeline
![Data Flow](visualizations/DATA_FLOW_DIAGRAM.png)
```

---

## 🔍 What Each Diagram Answers

### ARCHITECTURE_DIAGRAM answers:
- ❓ What are the main components?
- ❓ How are they connected?
- ❓ What are the inputs/outputs?
- ❓ Which models are used?

### FLOW_DIAGRAM answers:
- ❓ What happens step-by-step?
- ❓ Where does data split occur?
- ❓ When are models trained/tested?
- ❓ What files are generated?

### DATA_FLOW_DIAGRAM answers:
- ❓ How does data transform?
- ❓ What are the shapes at each stage?
- ❓ How is scaling applied?
- ❓ Where is data leakage prevented?

---

## ✅ Verification

All 3 diagrams include:
- ✅ Complete data flow from input to output
- ✅ All 6 models shown
- ✅ Top 2 models highlighted (GB & XGBoost)
- ✅ 80/20 split clearly marked
- ✅ Training vs testing separation
- ✅ StandardScaler placement
- ✅ Final metrics displayed
- ✅ File outputs listed
- ✅ Color-coded for clarity
- ✅ Professional quality (300 DPI)

---

## 🎓 Technical Details

### Data Pipeline:
```
361 samples → Parse → (361, 4) features
    ↓
Split 80/20
    ↓
288 train, 73 test
    ↓
Scale (fit on train, transform both)
    ↓
Train 6 models → Predict → Evaluate
    ↓
Select top 2 (GB, XGBoost)
    ↓
Test on 73 samples
    ↓
GB: 0.9426, XGB: 0.9420
```

### File Sizes:
- ARCHITECTURE_DIAGRAM: 475 KB (16x12 inches)
- FLOW_DIAGRAM: 713 KB (14x18 inches)  
- DATA_FLOW_DIAGRAM: 469 KB (16x10 inches)

### Color Scheme:
- **Blue**: Data input/processing
- **Green**: Training data/processes
- **Orange**: Testing data/processes
- **Red**: Model training
- **Purple**: Selection/storage
- **Gold**: Best models
- **Gray**: Outputs

---

## 📊 Summary

**3 diagrams = Complete visualization**:

1. **ARCHITECTURE**: "What is the system?"
2. **FLOW**: "How does it work?"
3. **DATA FLOW**: "How does data transform?"

**Total visual coverage**: From raw CSV to final predictions!

---

**Last Updated**: November 10, 2025  
**Resolution**: 300 DPI (print quality)  
**Total Size**: 1.6 MB (all 3 diagrams)
