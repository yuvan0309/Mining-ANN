"""Vercel serverless function for FoS prediction - Simplified."""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from pathlib import Path

# Global variables
MODEL = None
MODEL_INFO = None

def load_svm_model():
    """Load only the SVM model (most compatible with serverless)."""
    global MODEL, MODEL_INFO
    
    if MODEL is not None:
        return
    
    try:
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        
        # Load SVM model (best model, pure Python)
        MODEL = joblib.load(models_dir / "svm.joblib")
        
        # Load performance data
        with open(models_dir / "model_performance.json", 'r') as f:
            performance_data = json.load(f)
        
        # Get SVM info
        for model_data in performance_data:
            if model_data['model'] == 'svm':
                MODEL_INFO = {
                    'name': 'SVM',
                    'r2': model_data['r2'],
                    'rmse': model_data['rmse'],
                    'mae': model_data['mae']
                }
                break
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
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
        """Handle POST requests."""
        try:
            # Load model
            load_svm_model()
            
            # Read request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            features = data.get('features', {})
            
            # Prepare features (30 total: 27 material + 3 metadata)
            feature_dict = {
                'mine_label': 'Bicholim Mine A',
                'season': 'pre_monsoon',
                'point_index': 1
            }
            
            # Material properties
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
            
            for name in feature_names:
                try:
                    feature_dict[name] = float(features.get(name, 0.0))
                except (ValueError, TypeError):
                    feature_dict[name] = 0.0
            
            # Create DataFrame
            X = pd.DataFrame([feature_dict])
            
            # Predict
            prediction = MODEL.predict(X)[0]
            
            # Response
            response_data = {
                'success': True,
                'prediction': float(prediction),
                'model_used': MODEL_INFO['name'],
                'model_r2': float(MODEL_INFO['r2']),
                'model_rmse': float(MODEL_INFO['rmse']),
                'model_mae': float(MODEL_INFO['mae'])
            }
            
            self.send_json_response(response_data)
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error: {error_trace}", file=sys.stderr)
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, 500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
