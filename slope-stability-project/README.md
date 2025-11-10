# Slope Stability Prediction Project

Complete machine learning project for predicting Factor of Safety (FoS) in slope stability analysis using Bishop's Simplified Method.

## 📁 Project Structure

```
slope-stability-project/
├── new/                           # ML Pipeline & Training
│   ├── data/                     # Dataset files
│   ├── models/                   # Trained models (.pkl files)
│   ├── results/                  # Training results
│   ├── visualizations/           # Generated charts & plots
│   ├── main_pipeline.py          # Main training pipeline
│   ├── train_models.py           # Model training scripts
│   ├── data_ingestion.py         # Data loading & preprocessing
│   ├── generate_visualizations.py # Visualization generation
│   ├── ARCHITECTURE_DIAGRAM.mmd  # System architecture
│   ├── FLOW_DIAGRAM.mmd          # Process flow diagram
│   ├── DATA_FLOW_DIAGRAM.mmd     # Data transformation flow
│   └── [Various documentation files]
│
└── web-app/                      # Web Application
    ├── backend/                  # Flask REST API
    │   ├── app.py               # Main API server
    │   ├── models/              # Model files (copied from ../new/models/)
    │   ├── requirements.txt     # Python dependencies
    │   └── venv/                # Virtual environment
    │
    ├── frontend/                 # Svelte UI
    │   ├── src/
    │   │   ├── App.svelte       # Main component
    │   │   ├── components/      # UI components
    │   │   │   ├── PredictionForm.svelte
    │   │   │   ├── ResultsDisplay.svelte
    │   │   │   └── ModelInfo.svelte
    │   │   ├── main.js
    │   │   └── app.css
    │   ├── package.json
    │   └── vite.config.js
    │
    ├── setup.sh                  # Automated setup script
    ├── run.sh                    # Launch script
    ├── README.md                 # Web app documentation
    ├── QUICKSTART.md             # Quick setup guide
    ├── SUMMARY.md                # Project summary
    └── PROJECT_COMPLETE.md       # Complete overview

```

## 🚀 Quick Start

### Option 1: Run Web Application

```bash
cd web-app

# Automated setup
./setup.sh

# Launch servers
./run.sh

# Open browser: http://localhost:3000
```

### Option 2: Train Models

```bash
cd new

# Install dependencies (if needed)
pip install -r requirements.txt

# Run full pipeline
python main_pipeline.py

# Or train specific models
python train_models.py
```

## 📊 Project Components

### 1. Machine Learning Pipeline (`new/`)
- **6 Models Trained**: Gradient Boosting, XGBoost, LightGBM, Random Forest, SVM, ANN
- **Top 2 Selected**: GB (R²=0.9426) and XGBoost (R²=0.9420)
- **Dataset**: 361 samples (80/20 train-test split)
- **Features**: 4 parameters (Cohesion, Friction Angle, Unit Weight, Ru)
- **Method**: Bishop's Simplified Method

### 2. Web Application (`web-app/`)
- **Backend**: Flask REST API serving trained models
- **Frontend**: Svelte interactive UI
- **Features**:
  - Real-time FoS prediction
  - Model comparison (GB vs XGBoost)
  - Color-coded safety status
  - 95% confidence intervals
  - Responsive design

## 🎯 Key Features

### ML Pipeline
✅ Automated data ingestion and preprocessing  
✅ 6 machine learning models with hyperparameter tuning  
✅ Comprehensive performance evaluation  
✅ Model persistence (joblib)  
✅ Visualization generation (PNG charts)  
✅ Mermaid architecture diagrams  

### Web Application
✅ RESTful API with CORS support  
✅ Interactive sliders with validation  
✅ Ru parameter highlighting  
✅ Safety classification (CRITICAL/WARNING/CAUTION/SAFE)  
✅ Model metrics display  
✅ Mobile-responsive UI  

## 📈 Model Performance

| Model | Test R² | Test RMSE | Test MAE | Rank |
|-------|---------|-----------|----------|------|
| **Gradient Boosting** | 0.9426 | 0.0834 | 0.0563 | 🥇 1st |
| **XGBoost** | 0.9420 | 0.0838 | 0.0597 | 🥈 2nd |
| LightGBM | 0.9395 | 0.0856 | 0.0619 | 3rd |
| Random Forest | 0.9369 | 0.0874 | 0.0641 | 4th |
| SVM | 0.8901 | 0.1153 | 0.0822 | 5th |
| ANN | 0.8813 | 0.1198 | 0.0884 | 6th |

## 🛠️ Tech Stack

### ML Pipeline
- Python 3.13
- scikit-learn 1.5.0
- XGBoost 2.0.0
- LightGBM 4.5.0
- NumPy, Pandas
- Matplotlib, Seaborn

### Web Application
- **Backend**: Flask 3.0.0, joblib
- **Frontend**: Svelte 4.2.0, Vite 5.0.0, Axios 1.6.0
- **Styling**: CSS custom properties, Inter font

## 📝 Documentation

### ML Pipeline Docs (`new/`)
- `README.md` - Main project overview
- `QUICK_START.md` - Getting started guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `FINE_TUNED_RESULTS.md` - Model tuning results
- `TEST_RESULTS_COMPARISON.md` - Performance comparison
- `OUTPUT_FILES_GUIDE.md` - Generated files reference
- `MERMAID_DIAGRAMS_README.md` - Diagram documentation

### Web App Docs (`web-app/`)
- `README.md` - Complete web app documentation
- `QUICKSTART.md` - 5-minute setup guide
- `SUMMARY.md` - Feature summary
- `PROJECT_COMPLETE.md` - Visual overview

## 🔗 Related Documentation

- [ML Pipeline README](./new/README.md)
- [Web App README](./web-app/README.md)
- [Architecture Diagram](./new/ARCHITECTURE_DIAGRAM.mmd)
- [Flow Diagram](./new/FLOW_DIAGRAM.mmd)

## 🎓 Research Context

This project implements machine learning models for slope stability analysis in mining applications, specifically for Goa's mining regions. The models predict Factor of Safety (FoS) using Bishop's Simplified Method with four key parameters.

## 📊 Input Parameters

1. **Cohesion (c)**: 0-100 kPa - Soil shear strength
2. **Friction Angle (φ)**: 0-45° - Internal friction
3. **Unit Weight (γ)**: 15-25 kN/m³ - Soil density
4. **Ru (Pore Pressure Ratio)**: 0-1 - Water pressure ratio

## 🎯 Safety Classification

| FoS Range | Status | Meaning |
|-----------|--------|---------|
| < 1.0 | CRITICAL | Slope failure likely |
| 1.0-1.3 | WARNING | Marginal stability |
| 1.3-1.5 | CAUTION | Requires monitoring |
| ≥ 1.5 | SAFE | Stable slope |

## 🚦 Project Status

✅ **ML Pipeline**: Complete - 6 models trained, top 2 selected  
✅ **Web Application**: Complete - Backend + Frontend deployed  
✅ **Documentation**: Complete - All READMEs and guides  
✅ **Diagrams**: Complete - Architecture, flow, and data diagrams  

## 👥 Usage

### For Researchers/Data Scientists
Navigate to `new/` directory for model training and experimentation.

### For End Users
Navigate to `web-app/` directory to run the prediction interface.

---

**Project**: Slope Stability Prediction using Machine Learning  
**Method**: Bishop's Simplified Method  
**Date**: November 2025  
**Status**: Production Ready ✅
