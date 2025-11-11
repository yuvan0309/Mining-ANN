"""
Generate Histogram Plots with Normal Distribution (Bell Curve) for All Models
Creates publication-quality error distribution visualizations for all 6 models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Model list
ALL_MODELS = [
    'Gradient Boosting',
    'XGBoost',
    'LightGBM',
    'Random Forest',
    'SVM',
    'ANN'
]

# Color scheme for each model
MODEL_COLORS = {
    'Gradient Boosting': '#66c2a5',
    'XGBoost': '#fc8d62',
    'LightGBM': '#8da0cb',
    'Random Forest': '#e78ac3',
    'SVM': '#a6d854',
    'ANN': '#ffd92f'
}

def load_all_predictions():
    """Load predictions for all models"""
    print("Loading prediction files for all models...")
    
    predictions = {}
    
    # Load available prediction files
    for model in ALL_MODELS:
        model_key = model.lower().replace(' ', '_')
        file_path = f'models/test_predictions_{model_key}.csv'
        
        if os.path.exists(file_path):
            predictions[model] = pd.read_csv(file_path)
            print(f"  ✓ Loaded {model}")
        else:
            print(f"  ✗ File not found: {file_path}")
    
    # Also try the generic file
    if os.path.exists('models/test_predictions.csv'):
        generic_pred = pd.read_csv('models/test_predictions.csv')
        print(f"  ✓ Loaded generic predictions")
        
        # Check if it has multiple model columns
        if 'GB_Predictions' in generic_pred.columns:
            predictions['Gradient Boosting'] = pd.DataFrame({
                'Actual FoS': generic_pred['Actual FoS'],
                'Predicted FoS': generic_pred['GB_Predictions'],
                'Error': generic_pred['GB_Predictions'] - generic_pred['Actual FoS']
            })
        if 'XGB_Predictions' in generic_pred.columns:
            predictions['XGBoost'] = pd.DataFrame({
                'Actual FoS': generic_pred['Actual FoS'],
                'Predicted FoS': generic_pred['XGB_Predictions'],
                'Error': generic_pred['XGB_Predictions'] - generic_pred['Actual FoS']
            })
    
    return predictions

def plot_single_model_histogram(errors, model_name, output_path, color):
    """
    Create individual histogram with bell curve for one model
    
    Parameters:
    -----------
    errors : array-like
        Prediction errors
    model_name : str
        Name of the model
    output_path : str
        Path to save the plot
    color : str
        Color for the histogram
    """
    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    n = len(errors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    n_bins = 30
    counts, bins, patches = ax.hist(errors, bins=n_bins, 
                                     density=True, 
                                     alpha=0.7, 
                                     color=color,
                                     edgecolor='black',
                                     linewidth=1.2,
                                     label='Observed Errors')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(errors)
    
    # Plot bell curve (normal distribution)
    x = np.linspace(errors.min(), errors.max(), 1000)
    bell_curve = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, bell_curve, 'b-', linewidth=3, 
            label=f'Normal Distribution\n(Bell Curve)', alpha=0.9)
    
    # Plot mean line (red dashed)
    ax.axvline(mean_error, color='red', linestyle='--', 
               linewidth=2.5, label=f'Mean: {mean_error:.4f}')
    
    # Plot ±1 standard deviation lines (orange dashed)
    ax.axvline(mean_error + std_error, color='orange', 
               linestyle=':', linewidth=2.5, 
               label=f'±1 Std: {std_error:.4f}')
    ax.axvline(mean_error - std_error, color='orange', 
               linestyle=':', linewidth=2.5)
    
    # Plot zero error line (dark green)
    ax.axvline(0, color='darkgreen', linestyle='-', 
               linewidth=2.5, label='Zero Error', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Prediction Error (FoS units)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - Error Distribution with Bell Curve (n={n})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
              edgecolor='black', fancybox=True, shadow=True)
    
    # Add statistics text box
    # Check normality with Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(errors)
    normality = "Normal" if shapiro_p > 0.05 else "Non-normal"
    
    stats_text = (
        f'Statistics:\n'
        f'Mean Error: {mean_error:.4f}\n'
        f'Std Dev: {std_error:.4f}\n'
        f'Min Error: {errors.min():.4f}\n'
        f'Max Error: {errors.max():.4f}\n'
        f'Median: {np.median(errors):.4f}\n'
        f'Skewness: {stats.skew(errors):.4f}\n'
        f'Kurtosis: {stats.kurtosis(errors):.4f}\n'
        f'Normality: {normality}\n'
        f'Sample Size: {n}'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {pdf_path}")
    
    plt.close()
    
    return {
        'mean': mean_error,
        'std': std_error,
        'min': errors.min(),
        'max': errors.max(),
        'median': np.median(errors),
        'skewness': stats.skew(errors),
        'kurtosis': stats.kurtosis(errors),
        'normality_p': shapiro_p,
        'n': n
    }

def plot_all_models_comparison(all_errors, output_path):
    """
    Create a comprehensive comparison plot with all models (3x2 grid)
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    axes = axes.flatten()
    
    model_stats = {}
    
    for idx, (model_name, errors) in enumerate(all_errors.items()):
        ax = axes[idx]
        color = MODEL_COLORS.get(model_name, '#66c2a5')
        
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        n = len(errors)
        
        # Plot histogram
        n_bins = 25
        ax.hist(errors, bins=n_bins, 
                density=True, 
                alpha=0.7, 
                color=color,
                edgecolor='black',
                linewidth=1,
                label='Observed Errors')
        
        # Fit and plot bell curve
        mu, sigma = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 1000)
        bell_curve = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, bell_curve, 'b-', linewidth=2.5, 
                label=f'Bell Curve', alpha=0.9)
        
        # Plot lines
        ax.axvline(mean_error, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_error:.4f}')
        ax.axvline(mean_error + std_error, color='orange', 
                   linestyle=':', linewidth=2)
        ax.axvline(mean_error - std_error, color='orange', 
                   linestyle=':', linewidth=2, 
                   label=f'±1σ: {std_error:.4f}')
        ax.axvline(0, color='darkgreen', linestyle='-', 
                   linewidth=2, label='Zero', alpha=0.7)
        
        # Styling
        ax.set_xlabel('Prediction Error (FoS)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} (n={n})', 
                     fontsize=14, fontweight='bold', pad=10)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
        
        # Store stats
        model_stats[model_name] = {
            'mean': mean_error,
            'std': std_error,
            'n': n
        }
    
    plt.suptitle('Error Distribution Comparison - All Models', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved comparison: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved comparison: {pdf_path}")
    
    plt.close()
    
    return model_stats

def generate_statistics_table(all_stats, output_path):
    """Generate comprehensive statistics table"""
    
    stats_data = []
    for model_name, stats in all_stats.items():
        stats_data.append({
            'Model': model_name,
            'Mean Error': f"{stats['mean']:.4f}",
            'Std Dev': f"{stats['std']:.4f}",
            'Min Error': f"{stats['min']:.4f}",
            'Max Error': f"{stats['max']:.4f}",
            'Median': f"{stats['median']:.4f}",
            'Skewness': f"{stats['skewness']:.4f}",
            'Kurtosis': f"{stats['kurtosis']:.4f}",
            'Normality (p-value)': f"{stats['normality_p']:.4f}",
            'Sample Size': stats['n']
        })
    
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved statistics: {output_path}")
    
    # Also save as Excel
    excel_path = output_path.replace('.csv', '.xlsx')
    df.to_excel(excel_path, index=False, sheet_name='Error Statistics')
    print(f"  ✓ Saved statistics: {excel_path}")
    
    return df

def main():
    """Main execution function"""
    print("="*70)
    print("Generating Histogram Plots with Bell Curves for All Models")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = 'visualizations/all_models_histograms'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all predictions
    predictions = load_all_predictions()
    
    if not predictions:
        print("ERROR: No prediction files found!")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(predictions)} models with predictions")
    print(f"{'='*70}\n")
    
    # Generate individual plots
    print("Generating Individual Histogram Plots...")
    print("-" * 70)
    
    all_stats = {}
    all_errors = {}
    
    for model_name, pred_df in predictions.items():
        if 'Error' not in pred_df.columns:
            pred_df['Error'] = pred_df['Predicted FoS'] - pred_df['Actual FoS']
        
        errors = pred_df['Error'].values
        all_errors[model_name] = errors
        
        # Generate individual plot
        color = MODEL_COLORS.get(model_name, '#66c2a5')
        model_filename = model_name.lower().replace(' ', '_')
        output_path = f'{output_dir}/{model_filename}_histogram.png'
        
        stats = plot_single_model_histogram(errors, model_name, output_path, color)
        all_stats[model_name] = stats
    
    # Generate comparison plot
    print(f"\n{'='*70}")
    print("Generating Comparison Plot...")
    print("-" * 70)
    
    comparison_path = f'{output_dir}/all_models_comparison.png'
    plot_all_models_comparison(all_errors, comparison_path)
    
    # Generate statistics table
    print(f"\n{'='*70}")
    print("Generating Statistics Table...")
    print("-" * 70)
    
    stats_path = f'{output_dir}/error_statistics_all_models.csv'
    stats_df = generate_statistics_table(all_stats, stats_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print("="*70)
    print(stats_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("✓ All histogram plots generated successfully!")
    print(f"✓ Output directory: {output_dir}")
    print("="*70)
    print()
    print("Generated Files:")
    print(f"  • Individual histograms: {len(predictions)} models")
    print(f"  • Comparison plot: all_models_comparison.png")
    print(f"  • Statistics table: error_statistics_all_models.csv")
    print(f"  • All files available in PNG and PDF formats")
    print()

if __name__ == "__main__":
    main()
