"""Vercel serverless function for FoS prediction."""

from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)

# Global variables for models
MODELS = {}
MODEL_INFO = {}
BEST_MODEL = None

def load_models():
    """Load all trained models."""
    global MODELS, MODEL_INFO, BEST_MODEL
    
    if MODELS:  # Already loaded
        return
    
    # Get base directory
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    
    # Load performance data
    with open(models_dir / "model_performance.json", 'r') as f:
        performance_data = json.load(f)
    
    # Find best model
    sorted_models = sorted(performance_data, key=lambda x: x['r2'], reverse=True)
    BEST_MODEL = sorted_models[0]['model']
    
    # Model files
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

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Load models if not loaded
        load_models()
        
        data = request.json
        model_name = data.get('model', BEST_MODEL)
        features = data.get('features', {})
        
        # Validate model
        if model_name not in MODELS:
            return jsonify({
                'success': False,
                'error': f'Model "{model_name}" not found'
            }), 400
        
        # Prepare feature dictionary with metadata
        feature_dict = {
            'mine_label': 'Bicholim Mine A',
            'season': 'pre_monsoon',
            'point_index': 1
        }
        
        # Feature names (27 material properties)
        feature_names = [
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
        
        # Add material features
        for feature_name in feature_names:
            value = features.get(feature_name, 0.0)
            try:
                feature_dict[feature_name] = float(value)
            except (ValueError, TypeError):
                feature_dict[feature_name] = 0.0
        
        # Create DataFrame
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

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and their info."""
    try:
        load_models()
        
        return jsonify({
            'success': True,
            'models': MODEL_INFO,
            'best_model': BEST_MODEL
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS)
    })

# Vercel serverless handler
def handler(event, context):
    """Main handler for Vercel."""
    return app(event, context)

# For local testing
if __name__ == '__main__':
    app.run(debug=True)
