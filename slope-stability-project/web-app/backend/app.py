#!/usr/bin/env python3
"""
Flask API for FoS Prediction
Serves trained models via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load models and scaler
MODEL_DIR = Path(__file__).parent / 'models'

try:
    gb_model = joblib.load(MODEL_DIR / 'best_model_gradient_boosting.pkl')
    xgb_model = joblib.load(MODEL_DIR / 'best_model_xgboost.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please copy model files from ../new/models/ to ./backend/models/")
    gb_model = None
    xgb_model = None
    scaler = None

# Model metadata
MODEL_INFO = {
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'test_r2': 0.9426,
        'test_rmse': 0.0834,
        'test_mae': 0.0563,
        'training_r2': 0.9954,
        'overfitting_gap': '5.28%',
        'description': 'Best performing model with highest test accuracy'
    },
    'xgboost': {
        'name': 'XGBoost',
        'test_r2': 0.9420,
        'test_rmse': 0.0838,
        'test_mae': 0.0597,
        'training_r2': 0.9581,
        'overfitting_gap': '1.61%',
        'description': 'Excellent generalization with minimal overfitting'
    }
}


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'FoS Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make FoS prediction',
            '/models': 'GET - Get model information',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = gb_model is not None and xgb_model is not None and scaler is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'models not loaded',
        'models_loaded': models_loaded
    })


@app.route('/models')
def get_models():
    """Get model information"""
    return jsonify({
        'models': MODEL_INFO,
        'features': [
            'Cohesion (kPa)',
            'Friction Angle (degrees)',
            'Unit Weight (kN/m³)',
            'Ru (Pore Pressure Ratio)'
        ],
        'feature_ranges': {
            'cohesion': {'min': 0, 'max': 100, 'unit': 'kPa'},
            'friction_angle': {'min': 0, 'max': 45, 'unit': 'degrees'},
            'unit_weight': {'min': 15, 'max': 25, 'unit': 'kN/m³'},
            'ru': {'min': 0, 'max': 1, 'unit': 'ratio'}
        }
    })


def predict_multi_layer(data):
    """
    Handle multi-layer prediction by computing weighted average
    """
    try:
        layers = data['layers']
        model_choice = data.get('model', 'gradient_boosting')
        
        if not layers or len(layers) == 0:
            return jsonify({'error': 'No layers provided'}), 400
        
        # Select model
        if model_choice == 'xgboost':
            model = xgb_model
            model_name = 'XGBoost'
            model_metrics = MODEL_INFO['xgboost']
        else:
            model = gb_model
            model_name = 'Gradient Boosting'
            model_metrics = MODEL_INFO['gradient_boosting']
        
        # Store individual layer predictions
        layer_predictions = []
        total_weight = 0
        weighted_fos_sum = 0
        
        for idx, layer in enumerate(layers):
            # Validate layer data
            required_fields = ['name', 'cohesion', 'friction_angle', 'unit_weight', 'ru']
            for field in required_fields:
                if field not in layer:
                    return jsonify({
                        'error': f'Missing required field "{field}" in layer {idx + 1}'
                    }), 400
            
            # Extract layer properties
            cohesion = float(layer['cohesion'])
            friction_angle = float(layer['friction_angle'])
            unit_weight = float(layer['unit_weight'])
            ru = float(layer['ru'])
            
            # Validate ranges
            if not (0 <= cohesion <= 100):
                return jsonify({'error': f'Layer {layer["name"]}: Cohesion must be between 0 and 100 kPa'}), 400
            if not (0 <= friction_angle <= 45):
                return jsonify({'error': f'Layer {layer["name"]}: Friction angle must be between 0 and 45 degrees'}), 400
            if not (15 <= unit_weight <= 25):
                return jsonify({'error': f'Layer {layer["name"]}: Unit weight must be between 15 and 25 kN/m³'}), 400
            if not (0 <= ru <= 1):
                return jsonify({'error': f'Layer {layer["name"]}: Ru must be between 0 and 1'}), 400
            
            # Prepare features and predict
            features = np.array([[cohesion, friction_angle, unit_weight, ru]])
            features_scaled = scaler.transform(features)
            fos_prediction = float(model.predict(features_scaled)[0])
            
            # Store layer prediction
            layer_predictions.append({
                'name': layer['name'],
                'fos': round(fos_prediction, 4),
                'properties': {
                    'cohesion': cohesion,
                    'friction_angle': friction_angle,
                    'unit_weight': unit_weight,
                    'ru': ru
                }
            })
            
            # Calculate weighted average (using unit weight as weight factor)
            total_weight += unit_weight
            weighted_fos_sum += fos_prediction * unit_weight
        
        # Calculate overall FoS (weighted average)
        overall_fos = weighted_fos_sum / total_weight
        
        # Calculate confidence interval
        rmse = model_metrics['test_rmse']
        confidence_lower = float(max(0, overall_fos - 1.96 * rmse))
        confidence_upper = float(overall_fos + 1.96 * rmse)
        
        # Safety assessment
        if overall_fos < 1.0:
            safety_status = 'CRITICAL'
            safety_message = 'Slope is unstable - immediate action required'
            safety_color = 'red'
        elif overall_fos < 1.3:
            safety_status = 'WARNING'
            safety_message = 'Slope stability is marginal - review required'
            safety_color = 'orange'
        elif overall_fos < 1.5:
            safety_status = 'CAUTION'
            safety_message = 'Slope is stable but monitor conditions'
            safety_color = 'yellow'
        else:
            safety_status = 'SAFE'
            safety_message = 'Slope is stable'
            safety_color = 'green'
        
        return jsonify({
            'success': True,
            'prediction_type': 'multi-layer',
            'prediction': {
                'fos': round(overall_fos, 4),
                'confidence_interval': {
                    'lower': round(confidence_lower, 4),
                    'upper': round(confidence_upper, 4),
                    'level': '95%'
                }
            },
            'layers': layer_predictions,
            'calculation_method': 'Weighted average by unit weight',
            'safety': {
                'status': safety_status,
                'message': safety_message,
                'color': safety_color
            },
            'model': {
                'name': model_name,
                'r2_score': model_metrics['test_r2'],
                'rmse': model_metrics['test_rmse'],
                'mae': model_metrics['test_mae']
            }
        })
    
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input values',
            'message': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Multi-layer prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make FoS prediction
    
    Supports both single layer and multi-layer predictions:
    
    Single layer request:
    {
        "cohesion": float,
        "friction_angle": float,
        "unit_weight": float,
        "ru": float (optional, default=0),
        "model": "gradient_boosting" or "xgboost" (optional)
    }
    
    Multi-layer request:
    {
        "layers": [
            {
                "name": "Laterite",
                "cohesion": float,
                "friction_angle": float,
                "unit_weight": float,
                "ru": float
            },
            {
                "name": "Phyllitic Clay",
                "cohesion": float,
                "friction_angle": float,
                "unit_weight": float,
                "ru": float
            }
        ],
        "model": "gradient_boosting" or "xgboost" (optional)
    }
    """
    try:
        # Check if models are loaded
        if gb_model is None or xgb_model is None or scaler is None:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please ensure model files are in the models/ directory'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        # Check if multi-layer request
        if 'layers' in data:
            return predict_multi_layer(data)
        
        # Single layer prediction (original behavior)
        # Validate required fields
        required_fields = ['cohesion', 'friction_angle', 'unit_weight']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract features
        cohesion = float(data['cohesion'])
        friction_angle = float(data['friction_angle'])
        unit_weight = float(data['unit_weight'])
        ru = float(data.get('ru', 0.0))  # Default Ru = 0 if not provided
        model_choice = data.get('model', 'gradient_boosting')
        
        # Validate ranges
        if not (0 <= cohesion <= 100):
            return jsonify({'error': 'Cohesion must be between 0 and 100 kPa'}), 400
        if not (0 <= friction_angle <= 45):
            return jsonify({'error': 'Friction angle must be between 0 and 45 degrees'}), 400
        if not (15 <= unit_weight <= 25):
            return jsonify({'error': 'Unit weight must be between 15 and 25 kN/m³'}), 400
        if not (0 <= ru <= 1):
            return jsonify({'error': 'Ru must be between 0 and 1'}), 400
        
        # Prepare features
        features = np.array([[cohesion, friction_angle, unit_weight, ru]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Select model
        if model_choice == 'xgboost':
            model = xgb_model
            model_name = 'XGBoost'
            model_metrics = MODEL_INFO['xgboost']
        else:
            model = gb_model
            model_name = 'Gradient Boosting'
            model_metrics = MODEL_INFO['gradient_boosting']
        
        # Make prediction
        fos_prediction = float(model.predict(features_scaled)[0])
        
        # Calculate confidence interval (approximate using RMSE)
        rmse = model_metrics['test_rmse']
        confidence_lower = float(max(0, fos_prediction - 1.96 * rmse))
        confidence_upper = float(fos_prediction + 1.96 * rmse)
        
        # Safety assessment
        if fos_prediction < 1.0:
            safety_status = 'CRITICAL'
            safety_message = 'Slope is unstable - immediate action required'
            safety_color = 'red'
        elif fos_prediction < 1.3:
            safety_status = 'WARNING'
            safety_message = 'Slope stability is marginal - review required'
            safety_color = 'orange'
        elif fos_prediction < 1.5:
            safety_status = 'CAUTION'
            safety_message = 'Slope is stable but monitor conditions'
            safety_color = 'yellow'
        else:
            safety_status = 'SAFE'
            safety_message = 'Slope is stable'
            safety_color = 'green'
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': {
                'fos': round(fos_prediction, 4),
                'confidence_interval': {
                    'lower': round(confidence_lower, 4),
                    'upper': round(confidence_upper, 4),
                    'level': '95%'
                }
            },
            'safety': {
                'status': safety_status,
                'message': safety_message,
                'color': safety_color
            },
            'model': {
                'name': model_name,
                'r2_score': model_metrics['test_r2'],
                'rmse': model_metrics['test_rmse'],
                'mae': model_metrics['test_mae']
            },
            'inputs': {
                'cohesion': cohesion,
                'friction_angle': friction_angle,
                'unit_weight': unit_weight,
                'ru': ru
            }
        })
    
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input values',
            'message': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FoS PREDICTION API SERVER")
    print("="*60)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /         - API information")
    print("  GET  /health   - Health check")
    print("  GET  /models   - Model information")
    print("  POST /predict  - Make prediction")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
