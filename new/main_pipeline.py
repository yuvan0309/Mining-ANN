#!/usr/bin/env python3
"""
Main pipeline for FoS prediction with Ru (pore pressure ratio) incorporation.

This pipeline:
1. Loads data with Ru values
2. Splits into 80% training, 20% testing
3. Trains ALL models on 80% data
4. Selects best performing model
5. Tests ONLY best model on 20% data
6. Generates visualizations

Based on Bishop's Simplified Method with pore pressure.
"""

from pathlib import Path
from data_ingestion import load_and_prepare_data, train_test_split_data
from train_models import FoSModelTrainer
from generate_visualizations import create_all_visualizations


def run_pipeline(csv_path, include_ru=True, test_size=0.2, random_state=42):
    """
    Run the complete ML pipeline for FoS prediction.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to Overall Data.csv
    include_ru : bool
        Whether to include Ru values as features
    test_size : float
        Proportion of test data (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
    """
    
    print("\n" + "="*80)
    print("FoS PREDICTION PIPELINE WITH RU (PORE PRESSURE RATIO)")
    print("Based on Bishop's Simplified Method")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n📂 STEP 1: Loading Data...")
    X, y, df = load_and_prepare_data(csv_path, include_ru=include_ru)
    
    # Step 2: Split data
    print("\n✂️  STEP 2: Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 3: Train all models on 80% training data
    print("\n🔧 STEP 3: Training All Models...")
    trainer = FoSModelTrainer(X_train, y_train, X_test, y_test)
    training_results = trainer.train_all_models()
    
    # Step 4: Test top 2 models (Gradient Boosting & XGBoost) on 20% test data
    print("\n🧪 STEP 4: Testing Top Models (GB & XGBoost)...")
    test_results = trainer.test_best_models()
    
    # Step 5: Save models and results
    print("\n💾 STEP 5: Saving Models and Results...")
    models_dir = Path(__file__).parent / "models"
    trainer.save_models_and_results(models_dir)
    
    # Step 6: Generate visualizations
    print("\n📊 STEP 6: Generating Visualizations...")
    results_json = models_dir / "results_summary.json"
    viz_dir = Path(__file__).parent / "visualizations"
    create_all_visualizations(results_json, viz_dir)
    
    # Summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📋 Summary:")
    print(f"  • Total Samples: {len(X)}")
    print(f"  • Training Samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"  • Testing Samples: {len(X_test)} ({test_size*100:.0f}%)")
    print(f"  • Features: {list(X.columns)}")
    print(f"  • Models Trained: {len(training_results)}")
    print(f"  • Models Tested: {len(test_results)} (Gradient Boosting & XGBoost)")
    print(f"\n  📊 Test Results:")
    for model_name, results in sorted(test_results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"    • {model_name}:")
        print(f"      - R²: {results['r2']:.4f}")
        print(f"      - RMSE: {results['rmse']:.4f}")
        print(f"      - MAE: {results['mae']:.4f}")
    print(f"\n📁 Outputs:")
    print(f"  • Models saved: {models_dir}")
    print(f"  • Visualizations saved: {viz_dir}")
    print("="*80 + "\n")
    
    return trainer, training_results, test_results


if __name__ == "__main__":
    # Set paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "new" / "data" / "Overall Data.csv"
    
    # Check if data file exists
    if not csv_path.exists():
        print(f"❌ Error: Data file not found at {csv_path}")
        print(f"   Please ensure 'Overall Data.csv' is in the 'new/data' directory")
        exit(1)
    
    # Run pipeline with Ru values included
    print("🚀 Starting pipeline with Ru (pore pressure ratio) values...")
    trainer, training_results, test_results = run_pipeline(
        csv_path=csv_path,
        include_ru=True,
        test_size=0.2,
        random_state=42
    )
    
    print("✨ All done! Check the 'models' and 'visualizations' directories for outputs.")
