"""Vercel serverless function for FoS prediction.""""""Vercel serverless function for FoS prediction."""



from http.server import BaseHTTPRequestHandlerfrom flask import Flask, request, jsonify

from pathlib import Pathfrom pathlib import Path

import joblibimport joblib

import pandas as pdimport pandas as pd

import jsonimport json

import osimport os



# Global variables for modelsapp = Flask(__name__)

MODELS = {}

MODEL_INFO = {}# Global variables for models

BEST_MODEL = NoneMODELS = {}

MODEL_INFO = {}

def load_models():BEST_MODEL = None

    """Load all trained models."""

    global MODELS, MODEL_INFO, BEST_MODELdef load_models():

        """Load all trained models."""

    if MODELS:  # Already loaded    global MODELS, MODEL_INFO, BEST_MODEL

        return    

        if MODELS:  # Already loaded

    # Get base directory        return

    base_dir = Path(__file__).resolve().parent.parent    

    models_dir = base_dir / "models"    # Get base directory

        base_dir = Path(__file__).resolve().parent.parent

    # Load performance data    models_dir = base_dir / "models"

    with open(models_dir / "model_performance.json", 'r') as f:    

        performance_data = json.load(f)    # Load performance data

        with open(models_dir / "model_performance.json", 'r') as f:

    # Find best model        performance_data = json.load(f)

    sorted_models = sorted(performance_data, key=lambda x: x['r2'], reverse=True)    

    BEST_MODEL = sorted_models[0]['model']    # Find best model

        sorted_models = sorted(performance_data, key=lambda x: x['r2'], reverse=True)

    # Model files    BEST_MODEL = sorted_models[0]['model']

    model_files = {    

        'svm': 'svm.joblib',    # Model files

        'random_forest': 'random_forest.joblib',    model_files = {

        'xgboost': 'xgboost.joblib',        'svm': 'svm.joblib',

        'lightgbm': 'lightgbm.joblib',        'random_forest': 'random_forest.joblib',

        'ann_mlp': 'ann_mlp.joblib'        'xgboost': 'xgboost.joblib',

    }        'lightgbm': 'lightgbm.joblib',

            'ann_mlp': 'ann_mlp.joblib'

    # Load each model    }

    for model_data in performance_data:    

        model_name = model_data['model']    # Load each model

        if model_name in model_files:    for model_data in performance_data:

            model_path = models_dir / model_files[model_name]        model_name = model_data['model']

            if model_path.exists():        if model_name in model_files:

                MODELS[model_name] = joblib.load(model_path)            model_path = models_dir / model_files[model_name]

                MODEL_INFO[model_name] = {            if model_path.exists():

                    'name': model_name.replace('_', ' ').title(),                MODELS[model_name] = joblib.load(model_path)

                    'r2': model_data['r2'],                MODEL_INFO[model_name] = {

                    'rmse': model_data['rmse'],                    'name': model_name.replace('_', ' ').title(),

                    'mae': model_data['mae']                    'r2': model_data['r2'],

                }                    'rmse': model_data['rmse'],

                    'mae': model_data['mae']

class handler(BaseHTTPRequestHandler):                }

    def do_POST(self):

        """Handle POST requests for predictions."""@app.route('/api/predict', methods=['POST'])

        try:def predict():

            # Load models    """Handle prediction requests."""

            load_models()    try:

                    # Load models if not loaded

            # Read request body        load_models()

            content_length = int(self.headers['Content-Length'])        

            body = self.rfile.read(content_length)        data = request.json

            data = json.loads(body.decode('utf-8'))        model_name = data.get('model', BEST_MODEL)

                    features = data.get('features', {})

            model_name = data.get('model', BEST_MODEL)        

            features = data.get('features', {})        # Validate model

                    if model_name not in MODELS:

            # Validate model            return jsonify({

            if model_name not in MODELS:                'success': False,

                self.send_error_response({                'error': f'Model "{model_name}" not found'

                    'success': False,            }), 400

                    'error': f'Model "{model_name}" not found'        

                }, 400)        # Prepare feature dictionary with metadata

                return        feature_dict = {

                        'mine_label': 'Bicholim Mine A',

            # Prepare feature dictionary with metadata            'season': 'pre_monsoon',

            feature_dict = {            'point_index': 1

                'mine_label': 'Bicholim Mine A',        }

                'season': 'pre_monsoon',        

                'point_index': 1        # Feature names (27 material properties)

            }        feature_names = [

                        'laterite_cohesion_kpa', 'laterite_friction_angle_deg', 'laterite_unit_weight_kn_per_m3',

            # Feature names (27 material properties)            'phyllitic_clay_cohesion_kpa', 'phyllitic_clay_friction_angle_deg', 'phyllitic_clay_unit_weight_kn_per_m3',

            feature_names = [            'lumpy_iron_ore_cohesion_kpa', 'lumpy_iron_ore_friction_angle_deg', 'lumpy_iron_ore_unit_weight_kn_per_m3',

                'laterite_cohesion_kpa', 'laterite_friction_angle_deg', 'laterite_unit_weight_kn_per_m3',            'limonitic_clay_cohesion_kpa', 'limonitic_clay_friction_angle_deg', 'limonitic_clay_unit_weight_kn_per_m3',

                'phyllitic_clay_cohesion_kpa', 'phyllitic_clay_friction_angle_deg', 'phyllitic_clay_unit_weight_kn_per_m3',            'manganiferous_clay_cohesion_kpa', 'manganiferous_clay_friction_angle_deg', 'manganiferous_clay_unit_weight_kn_per_m3',

                'lumpy_iron_ore_cohesion_kpa', 'lumpy_iron_ore_friction_angle_deg', 'lumpy_iron_ore_unit_weight_kn_per_m3',            'siliceous_clay_cohesion_kpa', 'siliceous_clay_friction_angle_deg', 'siliceous_clay_unit_weight_kn_per_m3',

                'limonitic_clay_cohesion_kpa', 'limonitic_clay_friction_angle_deg', 'limonitic_clay_unit_weight_kn_per_m3',            'bhq_cohesion_kpa', 'bhq_friction_angle_deg', 'bhq_unit_weight_kn_per_m3',

                'manganiferous_clay_cohesion_kpa', 'manganiferous_clay_friction_angle_deg', 'manganiferous_clay_unit_weight_kn_per_m3',            'schist_cohesion_kpa', 'schist_friction_angle_deg', 'schist_unit_weight_kn_per_m3',

                'siliceous_clay_cohesion_kpa', 'siliceous_clay_friction_angle_deg', 'siliceous_clay_unit_weight_kn_per_m3',            'mean_cohesion_kpa', 'mean_friction_angle_deg', 'mean_unit_weight_kn_per_m3'

                'bhq_cohesion_kpa', 'bhq_friction_angle_deg', 'bhq_unit_weight_kn_per_m3',        ]

                'schist_cohesion_kpa', 'schist_friction_angle_deg', 'schist_unit_weight_kn_per_m3',        

                'mean_cohesion_kpa', 'mean_friction_angle_deg', 'mean_unit_weight_kn_per_m3'        # Add material features

            ]        for feature_name in feature_names:

                        value = features.get(feature_name, 0.0)

            # Add material features            try:

            for feature_name in feature_names:                feature_dict[feature_name] = float(value)

                value = features.get(feature_name, 0.0)            except (ValueError, TypeError):

                try:                feature_dict[feature_name] = 0.0

                    feature_dict[feature_name] = float(value)        

                except (ValueError, TypeError):        # Create DataFrame

                    feature_dict[feature_name] = 0.0        X = pd.DataFrame([feature_dict])

                    

            # Create DataFrame        # Make prediction

            X = pd.DataFrame([feature_dict])        model = MODELS[model_name]

                    prediction = model.predict(X)[0]

            # Make prediction        

            model = MODELS[model_name]        # Get model info

            prediction = model.predict(X)[0]        model_info = MODEL_INFO[model_name]

                    

            # Get model info        return jsonify({

            model_info = MODEL_INFO[model_name]            'success': True,

                        'prediction': float(prediction),

            # Send response            'model_used': model_info['name'],

            self.send_json_response({            'model_r2': float(model_info['r2']),

                'success': True,            'model_rmse': float(model_info['rmse']),

                'prediction': float(prediction),            'model_mae': float(model_info['mae'])

                'model_used': model_info['name'],        })

                'model_r2': float(model_info['r2']),        

                'model_rmse': float(model_info['rmse']),    except Exception as e:

                'model_mae': float(model_info['mae'])        return jsonify({

            })            'success': False,

                        'error': str(e)

        except Exception as e:        }), 500

            self.send_error_response({

                'success': False,@app.route('/api/models', methods=['GET'])

                'error': str(e)def get_models():

            }, 500)    """Get available models and their info."""

        try:

    def send_json_response(self, data, status=200):        load_models()

        """Send JSON response."""        

        self.send_response(status)        return jsonify({

        self.send_header('Content-type', 'application/json')            'success': True,

        self.send_header('Access-Control-Allow-Origin', '*')            'models': MODEL_INFO,

        self.end_headers()            'best_model': BEST_MODEL

        self.wfile.write(json.dumps(data).encode())        })

        except Exception as e:

    def send_error_response(self, data, status=500):        return jsonify({

        """Send error response."""            'success': False,

        self.send_json_response(data, status)            'error': str(e)

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
