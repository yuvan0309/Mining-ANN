"""Vercel serverless function - Optimized for serverless."""
from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Import at module level - fail fast if dependencies missing
try:
    import joblib
    import pandas as pd
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Global cache
MODELS = {}
MODEL_INFO = {}
INITIALIZED = False

def initialize():
    """Initialize models once."""
    global MODELS, MODEL_INFO, INITIALIZED
    
    if INITIALIZED:
        return True
    
    if not DEPS_AVAILABLE:
        raise Exception(f"Dependencies not available: {IMPORT_ERROR}")
    
    try:
        models_dir = BASE_DIR / "models"
        
        # Load performance data
        perf_file = models_dir / "model_performance.json"
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)
        
        # Load only SVM first (fastest, most compatible)
        try:
            svm_path = models_dir / "svm.joblib"
            MODELS['svm'] = joblib.load(svm_path)
            
            for perf in perf_data:
                if perf['model'] == 'svm':
                    MODEL_INFO['svm'] = {
                        'name': 'SVM',
                        'r2': perf['r2'],
                        'rmse': perf['rmse'],
                        'mae': perf['mae']
                    }
                    break
        except Exception as e:
            print(f"Failed to load SVM: {e}", file=sys.stderr)
        
        # Try other models (best effort)
        other_models = {
            'random_forest': 'random_forest.joblib',
            'ann_mlp': 'ann_mlp.joblib'
        }
        
        for model_name, filename in other_models.items():
            try:
                model_path = models_dir / filename
                MODELS[model_name] = joblib.load(model_path)
                
                for perf in perf_data:
                    if perf['model'] == model_name:
                        MODEL_INFO[model_name] = {
                            'name': model_name.replace('_', ' ').title(),
                            'r2': perf['r2'],
                            'rmse': perf['rmse'],
                            'mae': perf['mae']
                        }
                        break
            except:
                pass  # Skip if fails
        
        INITIALIZED = True
        return True
        
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return False

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
            # Initialize models
            if not initialize():
                raise Exception("Failed to initialize models")
            
            if not MODELS:
                raise Exception("No models available")
            
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            # Get model (use first available if requested not found)
            model_name = data.get('model', list(MODELS.keys())[0])
            if model_name not in MODELS:
                model_name = list(MODELS.keys())[0]
            
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
            
            # Predict
            X = pd.DataFrame([feature_dict])
            prediction = MODELS[model_name].predict(X)[0]
            
            # Response
            self.send_json({
                'success': True,
                'prediction': float(prediction),
                'model_used': MODEL_INFO[model_name]['name'],
                'model_r2': float(MODEL_INFO[model_name]['r2']),
                'model_rmse': float(MODEL_INFO[model_name]['rmse']),
                'model_mae': float(MODEL_INFO[model_name]['mae']),
                'available_models': list(MODELS.keys())
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({
                'success': False,
                'error': str(e),
                'type': type(e).__name__
            }, 500)
    
    def send_json(self, data, status=200):
        """Send JSON response - guaranteed."""
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except:
            pass  # Fail silently if already sent
