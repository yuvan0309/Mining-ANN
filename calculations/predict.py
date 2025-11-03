#!/usr/bin/env python3
"""Utility script for making FoS predictions with trained models."""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_model(model_name="random_forest"):
    """Load a trained model.
    
    Args:
        model_name: One of 'random_forest', 'xgboost', 'ann_mlp', 'svm', 'lightgbm'
    
    Returns:
        Loaded model pipeline
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)


def predict_single_point(model, material_properties):
    """Make a single FoS prediction.
    
    Args:
        model: Trained model pipeline
        material_properties: Dict with material properties
    
    Returns:
        Predicted FoS value
    """
    df = pd.DataFrame([material_properties])
    
    # Calculate aggregated features if not present
    if 'mean_cohesion_kpa' not in df.columns:
        cohesion_cols = [col for col in df.columns if col.endswith('cohesion_kpa')]
        if cohesion_cols:
            df['mean_cohesion_kpa'] = df[cohesion_cols].mean(axis=1)
    
    if 'mean_friction_angle_deg' not in df.columns:
        friction_cols = [col for col in df.columns if col.endswith('friction_angle_deg')]
        if friction_cols:
            df['mean_friction_angle_deg'] = df[friction_cols].mean(axis=1)
    
    if 'mean_unit_weight_kn_per_m3' not in df.columns:
        unit_weight_cols = [col for col in df.columns if col.endswith('unit_weight_kn_per_m3')]
        if unit_weight_cols:
            df['mean_unit_weight_kn_per_m3'] = df[unit_weight_cols].mean(axis=1)
    
    prediction = model.predict(df)
    return float(prediction[0])


def example_usage():
    """Demonstrate how to use the prediction utility."""
    
    print("=" * 80)
    print("FoS Prediction Example - All Models Comparison")
    print("=" * 80)
    print()
    
    # Load all models
    print("Loading all trained models...")
    models = {
        'SVM': load_model("svm"),
        'Random Forest': load_model("random_forest"),
        'XGBoost': load_model("xgboost"),
        'LightGBM': load_model("lightgbm"),
        'ANN': load_model("ann_mlp"),
    }
    print("✓ All models loaded successfully")
    print()
    
    # Example point data (Point 1 from Bicholim Mine A post monsoon)
    example_data = {
        'laterite_cohesion_kpa': 19.54,
        'laterite_friction_angle_deg': 23.68,
        'laterite_unit_weight_kn_per_m3': 24.0,
        'phyllitic_clay_cohesion_kpa': 16.08,
        'phyllitic_clay_friction_angle_deg': 16.85,
        'phyllitic_clay_unit_weight_kn_per_m3': 19.23,
        'lumpy_iron_ore_cohesion_kpa': 31.40,
        'lumpy_iron_ore_friction_angle_deg': 31.43,
        'lumpy_iron_ore_unit_weight_kn_per_m3': 27.91,
        'limonitic_clay_cohesion_kpa': 14.28,
        'limonitic_clay_friction_angle_deg': 20.54,
        'limonitic_clay_unit_weight_kn_per_m3': 18.06,
        'manganiferous_clay_cohesion_kpa': 18.44,
        'manganiferous_clay_friction_angle_deg': 23.52,
        'manganiferous_clay_unit_weight_kn_per_m3': 17.31,
        'siliceous_clay_cohesion_kpa': 18.58,
        'siliceous_clay_friction_angle_deg': 21.06,
        'siliceous_clay_unit_weight_kn_per_m3': 17.24,
        'bhq_cohesion_kpa': 37.94,
        'bhq_friction_angle_deg': 35.51,
        'bhq_unit_weight_kn_per_m3': 29.21,
        'schist_cohesion_kpa': 14.68,
        'schist_friction_angle_deg': 26.03,
        'schist_unit_weight_kn_per_m3': 22.87,
        'mine_label': 'Bicholim Mine A post monsoon',
        'season': 'postmonsoon',
        'point_index': 1,
    }
    
    actual_fos = 1.16  # Known value from CSV
    
    # Make predictions with all models
    print("Making predictions for example data point...")
    print(f"\n{'Model':<20} {'Prediction':>12} {'Error':>12} {'Relative Error':>15}")
    print("-" * 60)
    
    for model_name, model in models.items():
        predicted_fos = predict_single_point(model, example_data)
        error = abs(predicted_fos - actual_fos)
        rel_error = error / actual_fos * 100
        print(f"{model_name:<20} {predicted_fos:>12.4f} {error:>12.4f} {rel_error:>14.2f}%")
    
    print("-" * 60)
    print(f"{'Actual FoS (ground truth)':<20} {actual_fos:>12.4f}")
    print()
    
    # Show best model prediction
    best_model_name = 'SVM'  # Based on training results
    best_prediction = predict_single_point(models[best_model_name], example_data)
    
    # Interpretation
    print("=" * 80)
    print("FoS Safety Interpretation (Best Model: SVM):")
    print("-" * 80)
    print(f"FoS = {best_prediction:.3f}")
    
    if best_prediction >= 1.5:
        status = "✅ VERY SAFE"
        color = "Safe with good margin"
    elif best_prediction >= 1.25:
        status = "✅ SAFE"
        color = "Acceptable safety margin"
    elif best_prediction >= 1.0:
        status = "⚠️  MARGINALLY SAFE"
        color = "Requires monitoring"
    else:
        status = "❌ UNSAFE"
        color = "Potential failure risk"
    
    print(f"Status: {status}")
    print(f"Comment: {color}")
    print("=" * 80)
    print()
    
    print("Note: FoS (Factor of Safety) guidelines:")
    print("  - FoS ≥ 1.5: Very safe, good safety margin")
    print("  - FoS = 1.25-1.5: Safe, acceptable margin")
    print("  - FoS = 1.0-1.25: Marginally safe, monitoring recommended")
    print("  - FoS < 1.0: Unsafe, slope failure likely")
    print()


def interactive_prediction():
    """Interactive mode for entering custom predictions."""
    
    print("=" * 80)
    print("Interactive FoS Prediction")
    print("=" * 80)
    print()
    
    model = load_model("random_forest")
    print("✓ Random Forest model loaded")
    print()
    
    print("Enter material properties (press Ctrl+C to exit):")
    print("-" * 80)
    
    try:
        # Simplified input - just the aggregated features
        mean_cohesion = float(input("Mean Cohesion (kPa): "))
        mean_friction = float(input("Mean Friction Angle (°): "))
        mean_unit_weight = float(input("Mean Unit Weight (kN/m³): "))
        season = input("Season (premonsoon/postmonsoon): ").lower()
        
        # Create minimal data point
        data = {
            'mean_cohesion_kpa': mean_cohesion,
            'mean_friction_angle_deg': mean_friction,
            'mean_unit_weight_kn_per_m3': mean_unit_weight,
            'season': season,
        }
        
        predicted_fos = predict_single_point(model, data)
        
        print()
        print("=" * 80)
        print(f"Predicted Factor of Safety: {predicted_fos:.4f}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prediction()
    else:
        example_usage()
