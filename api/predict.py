"""Vercel serverless function - All 5 models with fallback."""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Global cache
MODELS = {}
MODEL_INFO = {}
BEST_MODEL = None

def load_all_models():
    """Load all 5 models with individual error handling."""
    global MODELS, MODEL_INFO, BEST_MODEL
    
    if MODELS:
        return
    
    import joblib
    import pandas as pd
    
    models_dir = BASE_DIR / "models"
    
    # Load performance data
    with open(models_dir / "model_performance.json", 'r') as f:
        perf_data = json.load(f)
    
    # Find best model
    BEST_MODEL = sorted(perf_data, key=lambda x: x['r2'], reverse=True)[0]['model']
    
    # Try loading each model
    model_files = {
        'svm': 'svm.joblib',
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'lightgbm': 'lightgbm.joblib',
        'ann_mlp': 'ann_mlp.joblib'
    }
    
    for model_name, filename in model_files.items():
        try:
            model_path = models_dir / filename
            MODELS[model_name] = joblib.load(model_path)
            
            # Get performance info
            for perf in perf_data:
                if perf['model'] == model_name:
                    MODEL_INFO[model_name] = {
                        'name': model_name.replace('_', ' ').title(),
                        'r2': perf['r2'],
                        'rmse': perf['rmse'],
                        'mae': perf['mae']
                    }
                    break
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}", file=sys.stderr)
            continue
    
    if not MODELS:
        raise Exception("No models could be loaded!")

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle predictions."""
        try:
            import pandas as pd
            
            # Load models
            load_all_models()
            
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            # Get model selection
            model_name = data.get('model', BEST_MODEL)
            
            # Fallback if requested model not available
            if model_name not in MODELS:
                available = list(MODELS.keys())
                model_name = available[0] if available else BEST_MODEL
            
            # Prepare features
            features = data.get('features', {})
            feature_dict = {
                'mine_label': 'Bicholim Mine A',
                'season': 'pre_monsoon',
                'point_index': 1
            }
            
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
                except:
                    feature_dict[name] = 0.0
            
            # Create DataFrame and predict
            X = pd.DataFrame([feature_dict])
            model = MODELS[model_name]
            prediction = model.predict(X)[0]
            
            # Response
            response = {
                'success': True,
                'prediction': float(prediction),
                'model_used': MODEL_INFO[model_name]['name'],
                'model_r2': float(MODEL_INFO[model_name]['r2']),
                'model_rmse': float(MODEL_INFO[model_name]['rmse']),
                'model_mae': float(MODEL_INFO[model_name]['mae']),
                'available_models': list(MODELS.keys())
            }
            
            self.send_json(response)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({
                'success': False,
                'error': str(e),
                'available_models': list(MODELS.keys()) if MODELS else []
            }, 500)
    
    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
