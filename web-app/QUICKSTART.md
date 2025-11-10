# Quick Start Guide

Get the Slope Stability Predictor web application running in 5 minutes!

## Prerequisites Check

```bash
# Check Python (need 3.8+)
python3 --version

# Check Node.js (need 18+)
node --version

# Check npm
npm --version
```

## 🚀 Quick Setup (Copy & Paste)

### 1. Copy Models (30 seconds)

```bash
cd "/home/inanotherlife/Mining ANN/web-app/backend/"
mkdir -p models
cp ../../new/models/gradient_boosting_model.pkl models/
cp ../../new/models/xgboost_model.pkl models/
cp ../../new/models/scaler.pkl models/
ls -lh models/  # Verify 3 files copied
```

### 2. Setup Backend (2 minutes)

```bash
cd "/home/inanotherlife/Mining ANN/web-app/backend/"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Setup Frontend (2 minutes)

```bash
cd "/home/inanotherlife/Mining ANN/web-app/frontend/"
npm install
```

## 🏃 Run the App

### Terminal 1: Backend

```bash
cd "/home/inanotherlife/Mining ANN/web-app/backend/"
source venv/bin/activate
python app.py
```

Wait for: `* Running on http://127.0.0.1:5000`

### Terminal 2: Frontend

```bash
cd "/home/inanotherlife/Mining ANN/web-app/frontend/"
npm run dev
```

Wait for: `➜  Local:   http://localhost:3000/`

### 🌐 Open Browser

Navigate to: **http://localhost:3000**

## 🎯 Quick Test

Try these values:
- **Cohesion:** 25 kPa
- **Friction Angle:** 30°
- **Unit Weight:** 18.5 kN/m³
- **Ru:** 0.3 (pore pressure ratio - highlighted in blue!)
- **Model:** Gradient Boosting

Click **Predict Factor of Safety** → Should see FoS ≈ 1.45 (SAFE)

## 🔧 Common Issues

### Backend won't start
```bash
# Make sure you're in venv
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Models not found
```bash
# Check models directory
ls -lh backend/models/
# Should show 3 .pkl files
# If missing, repeat step 1
```

## 📊 Expected Results

When everything works:
- Backend shows "Backend ready! Available endpoints..."
- Frontend shows "Local: http://localhost:3000/"
- Browser loads clean UI with sliders
- Predictions return in < 1 second
- Ru parameter box has blue background

## 🎉 Success Checklist

- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] Browser loads http://localhost:3000
- [ ] Can adjust sliders
- [ ] Ru parameter is highlighted in blue
- [ ] Click predict returns FoS value
- [ ] Color-coded safety status appears
- [ ] Model info shows in sidebar

## Next Steps

1. Try different parameter combinations
2. Switch between GB and XGBoost models
3. Check confidence intervals
4. Review model metrics in sidebar
5. Test with extreme values

## 📖 Full Documentation

See `README.md` for:
- Detailed API documentation
- Troubleshooting guide
- Model performance metrics
- Architecture details
- Tech stack information

---

**Ready to predict slope stability! 🏔️**
