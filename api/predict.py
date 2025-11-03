"""Vercel serverless function for FoS prediction."""

from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path is set
import joblib
import pandas as pd
from pathlib import Path

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
    
    try:
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
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        raise

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for predictions."""
        try:
            # Load models
            load_models()
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({
                    'success': False,
                    'error': 'No data provided'
                }, 400)
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            model_name = data.get('model', BEST_MODEL)
            features = data.get('features', {})
            
            # Validate model
            if model_name not in MODELS:
                self.send_json_response({
                    'success': False,
                    'error': f'Model "{model_name}" not found. Available: {list(MODELS.keys())}'
                }, 400)
                return
            
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
            
            # Send response
            response_data = {
                'success': True,
                'prediction': float(prediction),
                'model_used': model_info['name'],
                'model_r2': float(model_info['r2']),
                'model_rmse': float(model_info['rmse']),
                'model_mae': float(model_info['mae'])
            }
            
            self.send_json_response(response_data)
            
        except json.JSONDecodeError as e:
            self.send_json_response({
                'success': False,
                'error': f'Invalid JSON: {str(e)}'
            }, 400)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in prediction: {error_trace}", file=sys.stderr)
            self.send_json_response({
                'success': False,
                'error': str(e),
                'trace': error_trace
            }, 500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_text = json.dumps(data)
        self.wfile.write(response_text.encode('utf-8'))
