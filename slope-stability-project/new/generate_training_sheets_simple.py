"""
Generate Training Output Sheets for All Models
Creates comprehensive Excel and CSV files with training and test results
"""

import pandas as pd
import json
import os
from datetime import datetime

def load_model_results():
    """Load all available model result files"""
    print("Loading model results...")
    
    # Load test results
    test_results = pd.read_csv('models/test_results.csv')
    print(f"  ✓ Loaded test results: {len(test_results)} models")
    
    # Load training results
    training_results = pd.read_csv('models/training_results.csv')
    print(f"  ✓ Loaded training results: {len(training_results)} models")
    
    # Load comparison summary
    comparison = pd.read_csv('models/comparison_summary.csv')
    print(f"  ✓ Loaded comparison summary")
    
    # Load JSON results if available
    try:
        with open('models/results_summary.json', 'r') as f:
            results_json = json.load(f)
        print(f"  ✓ Loaded JSON results")
    except:
        results_json = {}
        print(f"  ! JSON results not found")
    
    return training_results, test_results, comparison, results_json

def create_excel_output():
    """Create comprehensive Excel output"""
    
    training_results, test_results, comparison, results_json = load_model_results()
    
    # Sort test results by R² score to get ranking
    test_results_sorted = test_results.sort_values('R² Score', ascending=False).reset_index(drop=True)
    test_results_sorted['Rank'] = range(1, len(test_results_sorted) + 1)
    
    # Create output file
    output_file = 'models/COMPREHENSIVE_TRAINING_OUTPUT.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Overview
        overview_data = {
            'Project': ['Slope Stability Prediction using ML'],
            'Method': ["Bishop's Simplified Method"],
            'Date': [datetime.now().strftime('%B %Y')],
            'Total Models Trained': [len(test_results)],
            'Best Model (Test R²)': [test_results_sorted.iloc[0]['Model']],
            'Best Test R² Score': [f"{test_results_sorted.iloc[0]['R² Score']:.4f}"],
            'Dataset Size': [361],
            'Train Split': ['80% (288 samples)'],
            'Test Split': ['20% (73 samples)'],
            'Scaling Method': ['StandardScaler'],
            'Cross-Validation': ['5-fold'],
            'Random State': [42]
        }
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='Overview', index=False)
        
        # Sheet 2: Training Results
        training_results.to_excel(writer, sheet_name='Training Results', index=False)
        
        # Sheet 3: Test Results with Ranking
        test_results_sorted.to_excel(writer, sheet_name='Test Results (Ranked)', index=False)
        
        # Sheet 4: Comparison Data
        comparison.to_excel(writer, sheet_name='All Phases Comparison', index=False)
        
        # Sheet 5: Detailed Metrics with Overfitting Analysis
        detailed_metrics = []
        for idx, row in test_results_sorted.iterrows():
            model_name = row['Model']
            
            # Find corresponding training row
            train_row = training_results[training_results['Model'] == model_name].iloc[0]
            
            # Calculate overfitting
            overfitting = ((train_row['R² Score'] - row['R² Score']) / 
                          train_row['R² Score'] * 100)
            
            generalization = 'Excellent' if overfitting < 5 else 'Good' if overfitting < 10 else 'Fair' if overfitting < 15 else 'Poor'
            
            detailed_metrics.append({
                'Rank': row['Rank'],
                'Model': model_name,
                'Training R²': f"{train_row['R² Score']:.6f}",
                'Training RMSE': f"{train_row['RMSE']:.6f}",
                'Training MAE': f"{train_row['MAE']:.6f}",
                'Test R²': f"{row['R² Score']:.6f}",
                'Test RMSE': f"{row['RMSE']:.6f}",
                'Test MAE': f"{row['MAE']:.6f}",
                'Overfitting Gap (%)': f"{overfitting:.2f}",
                'Generalization': generalization
            })
        
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Sheet 6: Top Two Models Comparison
        top_two = test_results_sorted.head(2)
        top_two_data = []
        
        for idx, row in top_two.iterrows():
            model_name = row['Model']
            train_row = training_results[training_results['Model'] == model_name].iloc[0]
            
            top_two_data.append({
                'Metric': 'Rank',
                model_name: row['Rank']
            })
        
        # Build comparison table
        metrics = {
            'Test R²': [],
            'Test RMSE': [],
            'Test MAE': [],
            'Training R²': [],
            'Training RMSE': [],
            'Training MAE': []
        }
        
        model_names = []
        for idx, row in top_two.iterrows():
            model_name = row['Model']
            model_names.append(model_name)
            train_row = training_results[training_results['Model'] == model_name].iloc[0]
            
            metrics['Test R²'].append(f"{row['R² Score']:.6f}")
            metrics['Test RMSE'].append(f"{row['RMSE']:.6f}")
            metrics['Test MAE'].append(f"{row['MAE']:.6f}")
            metrics['Training R²'].append(f"{train_row['R² Score']:.6f}")
            metrics['Training RMSE'].append(f"{train_row['RMSE']:.6f}")
            metrics['Training MAE'].append(f"{train_row['MAE']:.6f}")
        
        best_two_df = pd.DataFrame(metrics, index=model_names).T
        best_two_df.insert(0, 'Metric', best_two_df.index)
        best_two_df.reset_index(drop=True, inplace=True)
        best_two_df.to_excel(writer, sheet_name='Top Two Models', index=False)
        
        print(f"  ✓ Saved Excel: {output_file}")
    
    return output_file, detailed_df, test_results_sorted

def export_individual_csvs(detailed_df, test_results_sorted):
    """Export individual CSV files"""
    
    print("\nExporting individual CSV files...")
    
    # 1. Ranked test results
    test_results_sorted.to_csv('models/test_results_ranked.csv', index=False)
    print("  ✓ test_results_ranked.csv")
    
    # 2. Detailed metrics
    detailed_df.to_csv('models/detailed_metrics_all_models.csv', index=False)
    print("  ✓ detailed_metrics_all_models.csv")
    
    # 3. Top models only
    top_models = test_results_sorted.head(2)
    top_models.to_csv('models/top_two_models.csv', index=False)
    print("  ✓ top_two_models.csv")

def generate_markdown_summary(detailed_df, test_results_sorted):
    """Generate markdown summary"""
    
    print("\nGenerating markdown summary...")
    
    markdown = """# Comprehensive Training Output Summary

## Project Overview
- **Project**: Slope Stability Prediction using Machine Learning
- **Method**: Bishop's Simplified Method
- **Date**: {}
- **Total Models**: {}
- **Best Model**: {} (R² = {:.4f})

## Model Rankings (by Test R²)

| Rank | Model | Test R² | Test RMSE | Test MAE |
|------|-------|---------|-----------|----------|
""".format(
        datetime.now().strftime('%B %Y'),
        len(test_results_sorted),
        test_results_sorted.iloc[0]['Model'],
        test_results_sorted.iloc[0]['R² Score']
    )
    
    for idx, row in test_results_sorted.iterrows():
        markdown += f"| {row['Rank']} | {row['Model']} | {row['R² Score']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |\n"
    
    markdown += """\n## Detailed Performance Analysis\n\n"""
    
    for idx, row in detailed_df.iterrows():
        markdown += f"""### {row['Rank']}. {row['Model']}\n
**Training Performance:**
- R² Score: {row['Training R²']}
- RMSE: {row['Training RMSE']}
- MAE: {row['Training MAE']}

**Test Performance:**
- R² Score: {row['Test R²']}
- RMSE: {row['Test RMSE']}
- MAE: {row['Test MAE']}

**Overfitting Analysis:**
- Gap: {row['Overfitting Gap (%)']}%
- Generalization: {row['Generalization']}

---

"""
    
    # Save markdown
    with open('models/TRAINING_SUMMARY.md', 'w') as f:
        f.write(markdown)
    
    print("  ✓ TRAINING_SUMMARY.md")
    
    return markdown

def main():
    """Main execution"""
    
    print("="*70)
    print("Generating Comprehensive Training Output Sheets")
    print("="*70)
    print()
    
    # Create Excel output
    excel_file, detailed_df, test_results_sorted = create_excel_output()
    
    # Export CSVs
    export_individual_csvs(detailed_df, test_results_sorted)
    
    # Generate markdown
    markdown_summary = generate_markdown_summary(detailed_df, test_results_sorted)
    
    print()
    print("="*70)
    print("✓ All training output sheets generated successfully!")
    print("="*70)
    print()
    print("Generated Files:")
    print(f"  • Excel: COMPREHENSIVE_TRAINING_OUTPUT.xlsx")
    print(f"  • CSV: test_results_ranked.csv")
    print(f"  • CSV: detailed_metrics_all_models.csv")
    print(f"  • CSV: top_two_models.csv")
    print(f"  • Markdown: TRAINING_SUMMARY.md")
    print()
    print("Top 3 Models:")
    for idx, row in test_results_sorted.head(3).iterrows():
        print(f"  {row['Rank']}. {row['Model']}: R² = {row['R² Score']:.4f}")
    print()

if __name__ == "__main__":
    main()
