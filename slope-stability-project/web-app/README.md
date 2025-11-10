# Slope Stability Predictor Web Application

A full-stack web application for predicting Factor of Safety (FoS) in slope stability analysis using machine learning models (Gradient Boosting and XGBoost).

## 📁 Project Structure

```
web-app/
├── backend/               # Flask API Server
│   ├── app.py            # Main API endpoints
│   ├── requirements.txt  # Python dependencies
│   └── models/           # Trained ML models (to be added)
│       ├── gradient_boosting_model.pkl
│       ├── xgboost_model.pkl
│       └── scaler.pkl
└── frontend/             # Svelte Application
    ├── src/
    │   ├── App.svelte
    │   ├── main.js
    │   ├── app.css
    │   └── components/
    │       ├── PredictionForm.svelte
    │       ├── ResultsDisplay.svelte
    │       └── ModelInfo.svelte
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+** (with pip)
- **Node.js 18+** (with npm)
- Trained model files from the main project

### Step 1: Copy Model Files

First, copy the trained models from your main project to the backend:

```bash
cd /home/inanotherlife/Mining\ ANN/web-app/backend/
mkdir -p models
cp ../../new/models/gradient_boosting_model.pkl models/
cp ../../new/models/xgboost_model.pkl models/
cp ../../new/models/scaler.pkl models/
```

Verify the files are copied:
```bash
ls -lh models/
# Should show:
# gradient_boosting_model.pkl
# xgboost_model.pkl
# scaler.pkl
```

### Step 2: Setup Backend (Flask API)

```bash
cd /home/inanotherlife/Mining\ ANN/web-app/backend/

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Setup Frontend (Svelte)

```bash
cd /home/inanotherlife/Mining\ ANN/web-app/frontend/

# Install dependencies
npm install
```

## 🏃 Running the Application

### Terminal 1: Start Backend Server

```bash
cd /home/inanotherlife/Mining\ ANN/web-app/backend/
source venv/bin/activate  # If using virtual environment
python app.py
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * Backend ready! Available endpoints:
   - GET  /          : API info
   - GET  /health    : Health check
   - GET  /models    : Model metadata
   - POST /predict   : FoS prediction
```

### Terminal 2: Start Frontend Server

```bash
cd /home/inanotherlife/Mining\ ANN/web-app/frontend/
npm run dev
```

Expected output:
```
  VITE v5.0.0  ready in 432 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### Access the Application

Open your browser and navigate to: **http://localhost:3000**

## 🧪 Testing the API

### Health Check
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": true}
```

### Get Model Information
```bash
curl http://localhost:5000/models
```

### Make a Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cohesion": 25.0,
    "friction_angle": 30.0,
    "unit_weight": 18.5,
    "ru": 0.3,
    "model": "gradient_boosting"
  }'
```

Expected response:
```json
{
  "success": true,
  "prediction": {
    "factor_of_safety": 1.458,
    "confidence_interval": {"lower": 1.374, "upper": 1.542},
    "safety_status": "SAFE",
    "input_parameters": {
      "cohesion": 25.0,
      "friction_angle": 30.0,
      "unit_weight": 18.5,
      "ru": 0.3
    },
    "model_used": "gradient_boosting",
    "model_metrics": {
      "test_r2": 0.9426,
      "test_rmse": 0.0834,
      "test_mae": 0.0563
    }
  }
}
```

## 📊 Using the Web Interface

1. **Input Parameters:**
   - Cohesion (c): 0-100 kPa
   - Friction Angle (φ): 0-45 degrees
   - Unit Weight (γ): 15-25 kN/m³
   - **Ru (Pore Pressure Ratio)**: 0-1 ⚠️ **Highlighted in blue**

2. **Select Model:**
   - Gradient Boosting (Recommended - Best accuracy)
   - XGBoost (Excellent generalization)

3. **View Results:**
   - Factor of Safety (FoS) value
   - Safety Status (CRITICAL/WARNING/CAUTION/SAFE)
   - Confidence Interval (95% CI)
   - Model Metrics (R², RMSE, MAE)
   - Input Parameters Recap

4. **Sidebar Information:**
   - Model performance metrics
   - Key features comparison
   - System information
   - Tech stack details

## 🎨 Safety Classification

| FoS Range | Status | Color | Icon | Meaning |
|-----------|--------|-------|------|---------|
| < 1.0 | CRITICAL | Red | ⛔ | Slope failure likely |
| 1.0 - 1.3 | WARNING | Orange | ⚠️ | Marginal stability |
| 1.3 - 1.5 | CAUTION | Yellow | ⚡ | Requires monitoring |
| ≥ 1.5 | SAFE | Green | ✅ | Stable slope |

## 🛠️ Tech Stack

### Backend
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.5.0** - ML models
- **XGBoost 2.0.0** - Gradient boosting
- **joblib** - Model serialization
- **flask-cors** - Cross-origin support

### Frontend
- **Svelte 4.2.0** - UI framework
- **Vite 5.0.0** - Build tool
- **Axios 1.6.0** - HTTP client
- **Inter Font** - Typography

## 📝 API Endpoints

### GET /
Returns API information and available endpoints.

### GET /health
Health check endpoint to verify backend status.

### GET /models
Returns metadata for both trained models (GB and XGBoost).

### POST /predict
Main prediction endpoint.

**Request Body:**
```json
{
  "cohesion": 25.0,          // 0-100 kPa
  "friction_angle": 30.0,    // 0-45 degrees
  "unit_weight": 18.5,       // 15-25 kN/m³
  "ru": 0.3,                 // 0-1 (pore pressure ratio)
  "model": "gradient_boosting"  // or "xgboost"
}
```

**Response:**
- `success`: Boolean
- `prediction`: Object with FoS, CI, status, metrics
- `error`: Error message (if failed)

## 🔧 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'flask'`
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError: models/gradient_boosting_model.pkl`
```bash
# Copy models from main project
cp ../../new/models/*.pkl backend/models/
```

**Problem:** Port 5000 already in use
```bash
# Change port in backend/app.py
# Line: app.run(debug=True, port=5001)
```

### Frontend Issues

**Problem:** `npm: command not found`
```bash
# Install Node.js
sudo apt update
sudo apt install nodejs npm
```

**Problem:** CORS errors in browser console
```bash
# Make sure backend is running on port 5000
# Check vite.config.js proxy settings
```

**Problem:** Connection refused to backend
```bash
# 1. Verify backend is running
curl http://localhost:5000/health

# 2. Check frontend proxy config
# frontend/vite.config.js should have:
# proxy: { '/api': 'http://localhost:5000' }
```

## 📈 Model Performance

### Gradient Boosting (Recommended)
- **Test R²:** 0.9426 (94.26% accuracy)
- **Test RMSE:** 0.0834
- **Test MAE:** 0.0563
- **Overfitting Gap:** 5.28%
- **Rank:** 1st (Best accuracy)

### XGBoost
- **Test R²:** 0.9420 (94.20% accuracy)
- **Test RMSE:** 0.0838
- **Test MAE:** 0.0597
- **Overfitting Gap:** 1.61%
- **Rank:** 2nd (Best generalization)

## 🎯 Features

- ✅ Real-time Factor of Safety prediction
- ✅ Two trained ML models (GB and XGBoost)
- ✅ Interactive sliders with validation
- ✅ **Ru parameter highlighted** (blue background)
- ✅ Color-coded safety status
- ✅ 95% confidence intervals
- ✅ Model performance metrics
- ✅ Responsive design (mobile-friendly)
- ✅ Modern UI with Inter font
- ✅ RESTful API with CORS support

## 📊 Input Parameters Explained

- **Cohesion (c):** Soil shear strength parameter (kPa)
- **Friction Angle (φ):** Internal friction angle of soil (degrees)
- **Unit Weight (γ):** Soil unit weight (kN/m³)
- **Ru (Pore Pressure Ratio):** ⚠️ **Additional parameter** - Ratio of pore water pressure to overburden stress (0-1)

## 🔗 Related Files

- Main ML Pipeline: `/home/inanotherlife/Mining ANN/new/`
- Architecture Diagrams: `/home/inanotherlife/Mining ANN/ARCHITECTURE_DIAGRAM.mmd`
- Flow Diagrams: `/home/inanotherlife/Mining ANN/FLOW_DIAGRAM.mmd`

---

**Built with ❤️ using Flask, Svelte, and Machine Learning**
