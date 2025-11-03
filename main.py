#!/usr/bin/env python3
"""Main execution script for FoS prediction model training and comparison."""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calculations.data_ingestion import load_dataset
from calculations.train_models import train_and_evaluate


def main():
    """Execute the full ML pipeline: data loading, training, and model comparison."""
    
    print("=" * 80)
    print("Factor of Safety (FoS) Prediction - ML Model Training Pipeline")
    print("=" * 80)
    print()
    
    base_dir = Path(__file__).resolve().parent
    
    # Step 1: Load and prepare dataset
    print("Step 1: Loading and preparing dataset...")
    print("-" * 80)
    try:
        dataset = load_dataset(base_dir)
        print(f"✓ Dataset loaded successfully: {len(dataset)} data points")
        print(f"✓ Features: {len(dataset.columns)} columns")
        print(f"✓ Target variable (FoS) range: [{dataset['fos'].min():.3f}, {dataset['fos'].max():.3f}]")
        print()
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1
    
    # Step 2: Train and evaluate models
    print("Step 2: Training models (Random Forest, XGBoost, ANN)...")
    print("-" * 80)
    try:
        reports = train_and_evaluate(base_dir)
        print("✓ All models trained successfully")
        print()
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Display comparison results
    print("Step 3: Model Performance Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'CV R² (mean)':<15}")
    print("-" * 80)
    
    sorted_reports = sorted(reports, key=lambda x: x['r2'], reverse=True)
    
    for i, report in enumerate(sorted_reports, 1):
        cv_str = f"{report['cv_r2_mean']:.4f}" if report.get('cv_r2_mean') is not None else "N/A"
        print(f"{report['model']:<20} {report['rmse']:<12.4f} {report['mae']:<12.4f} "
              f"{report['r2']:<12.4f} {cv_str:<15}")
    
    print("=" * 80)
    print()
    
    # Determine best model
    best_model = sorted_reports[0]
    print(f"🏆 Best Model: {best_model['model']}")
    print(f"   - R² Score: {best_model['r2']:.4f} (higher is better, max=1.0)")
    print(f"   - RMSE: {best_model['rmse']:.4f} (lower is better)")
    print(f"   - MAE: {best_model['mae']:.4f} (lower is better)")
    if best_model.get('cv_r2_mean') is not None:
        print(f"   - Cross-validation R²: {best_model['cv_r2_mean']:.4f} ± {best_model.get('cv_r2_std', 0):.4f}")
    print()
    
    print("=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"✓ Trained {len(reports)} models")
    print(f"✓ Model files saved in: {base_dir / 'models'}")
    print(f"✓ Performance metrics saved: {base_dir / 'models' / 'model_performance.json'}")
    print()
    print("Interpretation Guide:")
    print("  - R² (Coefficient of Determination): Proportion of variance explained (0-1)")
    print("    * >0.9 = Excellent, 0.7-0.9 = Good, 0.5-0.7 = Moderate, <0.5 = Poor")
    print("  - RMSE (Root Mean Square Error): Average prediction error magnitude")
    print("  - MAE (Mean Absolute Error): Average absolute prediction error")
    print("  - CV R²: Cross-validated R² score (more robust than hold-out R²)")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
