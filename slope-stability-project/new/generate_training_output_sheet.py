"""
Generate Comprehensive Training Output Sheet for All Models
Creates detailed Excel/CSV reports for ML model writeup
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import os

def load_model_results():
    """Load all model results and metadata"""
    print("Loading model results...")
    
    # Load training results
    training_results = pd.read_csv('models/training_results.csv')
    
    # Load test results  
    test_results = pd.read_csv('models/test_results.csv')
    
    # Load comparison summary
    comparison = pd.read_csv('models/comparison_summary.csv')
    
    # Load results summary JSON
    with open('models/results_summary.json', 'r') as f:
        results_json = json.load(f)
    
    return training_results, test_results, comparison, results_json

def create_detailed_training_sheet():
    """Create comprehensive training output sheet"""
    
    training_results, test_results, comparison, results_json = load_model_results()
    
    # Create Excel writer
    output_file = 'models/COMPREHENSIVE_TRAINING_OUTPUT.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Overview Summary
        overview_data = {
            'Project': ['Slope Stability Prediction using ML'],
            'Method': ["Bishop's Simplified Method"],
            'Date': ['November 2025'],
            'Total Models Trained': [6],
            'Best Model': ['Gradient Boosting'],
            'Dataset Size': [361],
            'Train Split': ['80% (288 samples)'],
            'Test Split': ['20% (73 samples)'],
            'Features': [4],
            'Feature Names': ['Cohesion, Friction Angle, Unit Weight, Ru'],
            'Target Variable': ['Factor of Safety (FoS)'],
            'Scaling Method': ['StandardScaler'],
            'Cross-Validation': ['5-fold'],
            'Random State': [42]
        }
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='Overview', index=False)
        
        # Sheet 2: Training Results (All Models)
        training_results.to_excel(writer, sheet_name='Training Results', index=False)
        
        # Sheet 3: Test Results (All Models)
        test_results.to_excel(writer, sheet_name='Test Results', index=False)
        
        # Sheet 4: Model Comparison
        comparison.to_excel(writer, sheet_name='Model Comparison', index=False)
        
        # Sheet 5: Detailed Performance Metrics
        detailed_metrics = []
        for model_name in ['Gradient Boosting', 'XGBoost', 'LightGBM', 
                          'Random Forest', 'SVM', 'ANN']:
            train_row = training_results[training_results['Model'] == model_name].iloc[0]
            test_row = test_results[test_results['Model'] == model_name].iloc[0]
            
            # Calculate overfitting
            overfitting = ((train_row['R² Score'] - test_row['R² Score']) / 
                          train_row['R² Score'] * 100)
            
            detailed_metrics.append({
                'Model': model_name,
                'Train R²': train_row['R² Score'],
                'Train RMSE': train_row['RMSE'],
                'Train MAE': train_row['MAE'],
                'Test R²': test_row['R² Score'],
                'Test RMSE': test_row['RMSE'],
                'Test MAE': test_row['MAE'],
                'Overfitting Gap (%)': round(overfitting, 2),
                'Generalization': 'Excellent' if overfitting < 5 else 'Good' if overfitting < 10 else 'Fair',
                'Ranking': comparison[comparison['Model'] == model_name]['Rank'].values[0]
            })
        
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Sheet 6: Best Two Models Analysis
        best_two_data = {
            'Metric': ['Model Name', 'Test R² Score', 'Test RMSE', 'Test MAE', 
                      'Training R² Score', 'Training RMSE', 'Training MAE',
                      'Overfitting Gap (%)', 'Rank', 'Recommendation'],
            'Gradient Boosting': [
                'Gradient Boosting',
                detailed_metrics[0]['Test R²'],
                detailed_metrics[0]['Test RMSE'],
                detailed_metrics[0]['Test MAE'],
                detailed_metrics[0]['Train R²'],
                detailed_metrics[0]['Train RMSE'],
                detailed_metrics[0]['Train MAE'],
                detailed_metrics[0]['Overfitting Gap (%)'],
                1,
                'Best overall accuracy'
            ],
            'XGBoost': [
                'XGBoost',
                detailed_metrics[1]['Test R²'],
                detailed_metrics[1]['Test RMSE'],
                detailed_metrics[1]['Test MAE'],
                detailed_metrics[1]['Train R²'],
                detailed_metrics[1]['Train RMSE'],
                detailed_metrics[1]['Train MAE'],
                detailed_metrics[1]['Overfitting Gap (%)'],
                2,
                'Best generalization'
            ]
        }
        best_two_df = pd.DataFrame(best_two_data)
        best_two_df.to_excel(writer, sheet_name='Best Two Models', index=False)
        
        # Sheet 7: Hyperparameters
        hyperparams_data = []
        if 'hyperparameters' in results_json:
            for model_name, params in results_json['hyperparameters'].items():
                for param_name, param_value in params.items():
                    hyperparams_data.append({
                        'Model': model_name,
                        'Parameter': param_name,
                        'Value': str(param_value)
                    })
        
        if hyperparams_data:
            hyperparams_df = pd.DataFrame(hyperparams_data)
            hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        
        # Sheet 8: Feature Importance (if available)
        try:
            # Try to load feature importances from models
            gb_model = joblib.load('models/best_model_gradient_boosting.pkl')
            xgb_model = joblib.load('models/best_model_xgboost.pkl')
            
            features = ['Cohesion', 'Friction Angle', 'Unit Weight', 'Ru']
            
            importance_data = {
                'Feature': features,
                'Gradient Boosting': gb_model.feature_importances_,
                'XGBoost': xgb_model.feature_importances_
            }
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Gradient Boosting', ascending=False)
            importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        except:
            print("Note: Could not load feature importances")
        
        # Sheet 9: Statistical Analysis
        stats_data = {
            'Metric': ['Mean R² (All Models)', 'Std Dev R²', 'Best R²', 'Worst R²',
                      'Mean RMSE (All Models)', 'Std Dev RMSE', 'Best RMSE', 'Worst RMSE',
                      'Mean MAE (All Models)', 'Std Dev MAE', 'Best MAE', 'Worst MAE'],
            'Training Set': [
                training_results['R² Score'].mean(),
                training_results['R² Score'].std(),
                training_results['R² Score'].max(),
                training_results['R² Score'].min(),
                training_results['RMSE'].mean(),
                training_results['RMSE'].std(),
                training_results['RMSE'].min(),
                training_results['RMSE'].max(),
                training_results['MAE'].mean(),
                training_results['MAE'].std(),
                training_results['MAE'].min(),
                training_results['MAE'].max()
            ],
            'Test Set': [
                test_results['R² Score'].mean(),
                test_results['R² Score'].std(),
                test_results['R² Score'].max(),
                test_results['R² Score'].min(),
                test_results['RMSE'].mean(),
                test_results['RMSE'].std(),
                test_results['RMSE'].min(),
                test_results['RMSE'].max(),
                test_results['MAE'].mean(),
                test_results['MAE'].std(),
                test_results['MAE'].min(),
                test_results['MAE'].max()
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistical Analysis', index=False)
    
    print(f"✓ Comprehensive training output saved: {output_file}")
    
    # Also create CSV version
    csv_dir = 'models/training_output_sheets'
    os.makedirs(csv_dir, exist_ok=True)
    
    overview_df.to_csv(f'{csv_dir}/01_overview.csv', index=False)
    training_results.to_csv(f'{csv_dir}/02_training_results.csv', index=False)
    test_results.to_csv(f'{csv_dir}/03_test_results.csv', index=False)
    comparison.to_csv(f'{csv_dir}/04_model_comparison.csv', index=False)
    detailed_df.to_csv(f'{csv_dir}/05_detailed_metrics.csv', index=False)
    best_two_df.to_csv(f'{csv_dir}/06_best_two_models.csv', index=False)
    stats_df.to_csv(f'{csv_dir}/09_statistical_analysis.csv', index=False)
    
    print(f"✓ Individual CSV sheets saved in: {csv_dir}/")
    
    return output_file

def create_markdown_summary():
    """Create a markdown summary for writeup"""
    
    training_results, test_results, comparison, results_json = load_model_results()
    
    markdown = """# ML Model Development, Training, Testing and Validation

## 1. Dataset Overview

- **Total Samples**: 361
- **Training Set**: 288 samples (80%)
- **Test Set**: 73 samples (20%)
- **Features**: 4 (Cohesion, Friction Angle, Unit Weight, Ru)
- **Target**: Factor of Safety (FoS)
- **Method**: Bishop's Simplified Method
- **Preprocessing**: StandardScaler normalization

## 2. Models Trained

Six machine learning models were trained and evaluated:

1. **Gradient Boosting Regressor**
2. **XGBoost**
3. **LightGBM**
4. **Random Forest**
5. **Support Vector Machine (SVM)**
6. **Artificial Neural Network (ANN)**

## 3. Training Results

| Model | Training R² | Training RMSE | Training MAE |
|-------|-------------|---------------|--------------|
"""
    
    for _, row in training_results.iterrows():
        markdown += f"| {row['Model']} | {row['R² Score']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |\n"
    
    markdown += """
## 4. Test Results

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
"""
    
    for _, row in test_results.iterrows():
        markdown += f"| {row['Model']} | {row['R² Score']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |\n"
    
    markdown += """
## 5. Model Comparison and Ranking

| Rank | Model | Test R² | Test RMSE | Test MAE |
|------|-------|---------|-----------|----------|
"""
    
    for _, row in comparison.iterrows():
        markdown += f"| {row['Rank']} | {row['Model']} | {row['Test R²']:.4f} | {row['Test RMSE']:.4f} | {row['Test MAE']:.4f} |\n"
    
    markdown += """
## 6. Best Model Selection

### Top 2 Models

#### 🥇 Gradient Boosting (Rank 1)
- **Test R²**: 0.9426 (94.26% variance explained)
- **Test RMSE**: 0.0834
- **Test MAE**: 0.0563
- **Strengths**: Highest test accuracy, robust predictions
- **Use Case**: Primary model for production deployment

#### 🥈 XGBoost (Rank 2)
- **Test R²**: 0.9420 (94.20% variance explained)
- **Test RMSE**: 0.0838
- **Test MAE**: 0.0597
- **Strengths**: Excellent generalization, minimal overfitting
- **Use Case**: Alternative model for cross-validation

## 7. Model Validation

### Overfitting Analysis

Both top models show minimal overfitting:
- **Gradient Boosting**: Training-Test gap < 5.3%
- **XGBoost**: Training-Test gap < 1.7%

### Cross-Validation

5-fold cross-validation was performed to ensure model stability and generalization capability.

## 8. Conclusion

The **Gradient Boosting** model was selected as the primary model due to:
1. Highest test accuracy (R² = 0.9426)
2. Lowest prediction errors (RMSE = 0.0834, MAE = 0.0563)
3. Strong generalization with minimal overfitting
4. Robust performance across all evaluation metrics

Both Gradient Boosting and XGBoost models are deployed in the production web application for ensemble predictions.

---

**Generated**: November 2025  
**Total Models Evaluated**: 6  
**Best Model**: Gradient Boosting  
**Validation Method**: Train-Test Split + 5-Fold CV
"""
    
    # Save markdown
    with open('models/TRAINING_VALIDATION_SUMMARY.md', 'w') as f:
        f.write(markdown)
    
    print("✓ Markdown summary saved: models/TRAINING_VALIDATION_SUMMARY.md")

def main():
    """Main execution"""
    print("="*70)
    print("Generating Comprehensive Training Output Sheets")
    print("="*70)
    print()
    
    # Create comprehensive Excel output
    excel_file = create_detailed_training_sheet()
    
    # Create markdown summary
    create_markdown_summary()
    
    print()
    print("="*70)
    print("✓ All training output sheets generated successfully!")
    print("="*70)
    print()
    print("Output Files:")
    print(f"  1. Excel: models/COMPREHENSIVE_TRAINING_OUTPUT.xlsx")
    print(f"  2. CSV Sheets: models/training_output_sheets/")
    print(f"  3. Markdown: models/TRAINING_VALIDATION_SUMMARY.md")
    print()

if __name__ == "__main__":
    main()
