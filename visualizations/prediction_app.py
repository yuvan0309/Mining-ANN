#!/usr/bin/env python3
"""Interactive web application for FoS prediction with model selection."""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = Flask(__name__)

# Global variables to store models and metadata
MODELS = {}
MODEL_INFO = {}
FEATURE_NAMES = []
BEST_MODEL = None

def load_models_and_metadata():
    """Load all trained models and their metadata."""
    global MODELS, MODEL_INFO, FEATURE_NAMES, BEST_MODEL
    
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    
    # Load model performance data
    with open(models_dir / "model_performance.json", 'r') as f:
        performance_data = json.load(f)
    
    # Sort to find best model
    sorted_models = sorted(performance_data, key=lambda x: x['r2'], reverse=True)
    BEST_MODEL = sorted_models[0]['model']
    
    # Model file mapping
    model_files = {
        'svm': 'svm.joblib',
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'lightgbm': 'lightgbm.joblib',
        'ann_mlp': 'ann_mlp.joblib'
    }
    
    # Load each model
    for model_data in performance_data:
        model_name = model_data['model']
        if model_name in model_files:
            model_path = models_dir / model_files[model_name]
            if model_path.exists():
                MODELS[model_name] = joblib.load(model_path)
                MODEL_INFO[model_name] = {
                    'name': model_name.replace('_', ' ').title(),
                    'r2': model_data['r2'],
                    'rmse': model_data['rmse'],
                    'mae': model_data['mae']
                }
    
    # Load feature names from dataset
    dataset_path = base_dir / "calculations" / "prepared_dataset.parquet"
    if dataset_path.exists():
        df = pd.read_parquet(dataset_path)
        FEATURE_NAMES = [col for col in df.columns 
                        if col not in ['fos', 'point_label', 'point_index', 
                                      'mine_label', 'season', 'source_file']]
    else:
        # Fallback feature names
        FEATURE_NAMES = [
            'laterite_cohesion_kpa', 'laterite_friction_angle_deg', 'laterite_unit_weight_kn_per_m3',
            'phyllitic_clay_cohesion_kpa', 'phyllitic_clay_friction_angle_deg', 'phyllitic_clay_unit_weight_kn_per_m3',
            'lumpy_iron_ore_cohesion_kpa', 'lumpy_iron_ore_friction_angle_deg', 'lumpy_iron_ore_unit_weight_kn_per_m3',
            'limonitic_clay_cohesion_kpa', 'limonitic_clay_friction_angle_deg', 'limonitic_clay_unit_weight_kn_per_m3',
            'manganiferous_clay_cohesion_kpa', 'manganiferous_clay_friction_angle_deg', 'manganiferous_clay_unit_weight_kn_per_m3',
            'siliceous_clay_cohesion_kpa', 'siliceous_clay_friction_angle_deg', 'siliceous_clay_unit_weight_kn_per_m3',
            'bhq_cohesion_kpa', 'bhq_friction_angle_deg', 'bhq_unit_weight_kn_per_m3',
            'schist_cohesion_kpa', 'schist_friction_angle_deg', 'schist_unit_weight_kn_per_m3',
            'mean_cohesion_kpa', 'mean_friction_angle_deg', 'mean_unit_weight_kn_per_m3'
        ]

@app.route('/')
def index():
    """Render the main prediction page."""
    return render_template('prediction.html', 
                          models=MODEL_INFO,
                          best_model=BEST_MODEL,
                          features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.json
        model_name = data.get('model', BEST_MODEL)
        features = data.get('features', {})
        
        # Validate model selection
        if model_name not in MODELS:
            return jsonify({
                'success': False,
                'error': f'Model "{model_name}" not found'
            }), 400
        
        # Prepare feature dictionary with all required features
        # Models expect: mine_label, season, point_index + 27 material features
        feature_dict = {
            'mine_label': 'Bicholim Mine A',  # Default value
            'season': 'pre_monsoon',           # Default value
            'point_index': 1                   # Default value
        }
        
        # Add material property features
        for feature_name in FEATURE_NAMES:
            value = features.get(feature_name, 0.0)
            try:
                feature_dict[feature_name] = float(value)
            except (ValueError, TypeError):
                feature_dict[feature_name] = 0.0
        
        # Create DataFrame with proper feature order
        X = pd.DataFrame([feature_dict])
        
        # Make prediction
        model = MODELS[model_name]
        prediction = model.predict(X)[0]
        
        # Get model info
        model_info = MODEL_INFO[model_name]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'model_used': model_info['name'],
            'model_r2': float(model_info['r2']),
            'model_rmse': float(model_info['rmse']),
            'model_mae': float(model_info['mae'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_sample_data')
def get_sample_data():
    """Return sample data from the dataset."""
    try:
        base_dir = Path(__file__).resolve().parent.parent
        dataset_path = base_dir / "calculations" / "prepared_dataset.parquet"
        
        if dataset_path.exists():
            df = pd.read_parquet(dataset_path)
            # Get a random sample
            sample = df.sample(n=1).iloc[0]
            
            sample_features = {}
            for feature_name in FEATURE_NAMES:
                if feature_name in sample:
                    value = sample[feature_name]
                    sample_features[feature_name] = float(value) if pd.notna(value) else 0.0
                else:
                    sample_features[feature_name] = 0.0
            
            actual_fos = float(sample['fos']) if 'fos' in sample else None
            
            return jsonify({
                'success': True,
                'features': sample_features,
                'actual_fos': actual_fos
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Load models at module level (before any requests)
print("Loading models and metadata...")
load_models_and_metadata()
print(f"Loaded {len(MODELS)} models")
print(f"Best model: {BEST_MODEL}")
print(f"Features: {len(FEATURE_NAMES)}")

if __name__ == '__main__':
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
