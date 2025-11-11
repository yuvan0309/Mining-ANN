"""
Generate Error Distribution Plots with Bell Curves for ML Models
Creates publication-quality error distribution visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_models_and_data():
    """Load trained models and test data"""
    print("Loading models and data...")
    
    # Load test predictions for both models
    gb_predictions = pd.read_csv('models/test_predictions_gradient_boosting.csv')
    xgb_predictions = pd.read_csv('models/test_predictions_xgboost.csv')
    
    return gb_predictions, xgb_predictions

def calculate_errors(y_true, y_pred):
    """Calculate prediction errors"""
    errors = y_pred - y_true
    return errors

def plot_error_distribution(errors, model_name, output_path):
    """
    Create error distribution plot with bell curve overlay
    
    Parameters:
    -----------
    errors : array-like
        Prediction errors
    model_name : str
        Name of the model
    output_path : str
        Path to save the plot
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
                                     color='#66c2a5',
                                     edgecolor='black',
                                     linewidth=1.2,
                                     label='Observed Errors')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(errors)
    
    # Plot bell curve
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
    stats_text = (
        f'Statistics:\n'
        f'Mean Error: {mean_error:.4f}\n'
        f'Std Dev: {std_error:.4f}\n'
        f'Min Error: {errors.min():.4f}\n'
        f'Max Error: {errors.max():.4f}\n'
        f'Median: {np.median(errors):.4f}\n'
        f'Sample Size: {n}'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()

def plot_combined_comparison(gb_errors, xgb_errors, output_path):
    """
    Create side-by-side comparison of both models
    
    Parameters:
    -----------
    gb_errors : array-like
        Gradient Boosting errors
    xgb_errors : array-like
        XGBoost errors
    output_path : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    models = [
        (gb_errors, 'Gradient Boosting', ax1),
        (xgb_errors, 'XGBoost', ax2)
    ]
    
    for errors, model_name, ax in models:
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        n = len(errors)
        
        # Plot histogram
        n_bins = 30
        ax.hist(errors, bins=n_bins, 
                density=True, 
                alpha=0.7, 
                color='#66c2a5',
                edgecolor='black',
                linewidth=1.2,
                label='Observed Errors')
        
        # Fit and plot bell curve
        mu, sigma = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 1000)
        bell_curve = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, bell_curve, 'b-', linewidth=3, 
                label=f'Normal Distribution\n(Bell Curve)', alpha=0.9)
        
        # Plot lines
        ax.axvline(mean_error, color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {mean_error:.4f}')
        ax.axvline(mean_error + std_error, color='orange', 
                   linestyle=':', linewidth=2.5, 
                   label=f'±1 Std: {std_error:.4f}')
        ax.axvline(mean_error - std_error, color='orange', 
                   linestyle=':', linewidth=2.5)
        ax.axvline(0, color='darkgreen', linestyle='-', 
                   linewidth=2.5, label='Zero Error', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Prediction Error (FoS units)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name} - Error Distribution (n={n})', 
                     fontsize=15, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                  edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()

def generate_error_statistics_table(gb_errors, xgb_errors, output_path):
    """Generate a comparison table of error statistics"""
    
    stats_data = {
        'Metric': ['Mean Error', 'Std Dev', 'Min Error', 'Max Error', 
                   'Median Error', '25th Percentile', '75th Percentile',
                   'Skewness', 'Kurtosis', 'Sample Size'],
        'Gradient Boosting': [
            f"{np.mean(gb_errors):.4f}",
            f"{np.std(gb_errors):.4f}",
            f"{gb_errors.min():.4f}",
            f"{gb_errors.max():.4f}",
            f"{np.median(gb_errors):.4f}",
            f"{np.percentile(gb_errors, 25):.4f}",
            f"{np.percentile(gb_errors, 75):.4f}",
            f"{stats.skew(gb_errors):.4f}",
            f"{stats.kurtosis(gb_errors):.4f}",
            f"{len(gb_errors)}"
        ],
        'XGBoost': [
            f"{np.mean(xgb_errors):.4f}",
            f"{np.std(xgb_errors):.4f}",
            f"{xgb_errors.min():.4f}",
            f"{xgb_errors.max():.4f}",
            f"{np.median(xgb_errors):.4f}",
            f"{np.percentile(xgb_errors, 25):.4f}",
            f"{np.percentile(xgb_errors, 75):.4f}",
            f"{stats.skew(xgb_errors):.4f}",
            f"{stats.kurtosis(xgb_errors):.4f}",
            f"{len(xgb_errors)}"
        ]
    }
    
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False)
    print(f"Saved error statistics: {output_path}")
    
    return df

def main():
    """Main execution function"""
    print("="*60)
    print("Error Distribution Analysis with Bell Curves")
    print("="*60)
    
    # Create output directory
    output_dir = 'visualizations/error_distributions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and data
    gb_predictions, xgb_predictions = load_models_and_data()
    
    # Get errors directly from the CSV files (they already contain error column)
    gb_errors = gb_predictions['Error'].values
    xgb_errors = xgb_predictions['Error'].values
    
    print(f"\nGradient Boosting:")
    print(f"  Mean Error: {np.mean(gb_errors):.4f}")
    print(f"  Std Dev: {np.std(gb_errors):.4f}")
    print(f"  Sample Size: {len(gb_errors)}")
    
    print(f"\nXGBoost:")
    print(f"  Mean Error: {np.mean(xgb_errors):.4f}")
    print(f"  Std Dev: {np.std(xgb_errors):.4f}")
    print(f"  Sample Size: {len(xgb_errors)}")
    
    # Generate individual plots
    print("\n" + "="*60)
    print("Generating Individual Error Distribution Plots...")
    print("="*60)
    
    plot_error_distribution(
        gb_errors, 
        'Gradient Boosting',
        f'{output_dir}/gradient_boosting_error_distribution.png'
    )
    
    plot_error_distribution(
        xgb_errors, 
        'XGBoost',
        f'{output_dir}/xgboost_error_distribution.png'
    )
    
    # Generate comparison plot
    print("\n" + "="*60)
    print("Generating Comparison Plot...")
    print("="*60)
    
    plot_combined_comparison(
        gb_errors, 
        xgb_errors,
        f'{output_dir}/both_models_error_distribution_comparison.png'
    )
    
    # Generate statistics table
    print("\n" + "="*60)
    print("Generating Statistics Table...")
    print("="*60)
    
    stats_df = generate_error_statistics_table(
        gb_errors,
        xgb_errors,
        f'{output_dir}/error_statistics_comparison.csv'
    )
    
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(stats_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ All error distribution plots generated successfully!")
    print(f"✓ Output directory: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
