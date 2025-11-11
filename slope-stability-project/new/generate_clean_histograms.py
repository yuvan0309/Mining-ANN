"""
Generate Clean Histogram Plots with Generalized Normal Distribution Curve
Removes outliers and creates publication-quality visualizations
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

# Color scheme
MODEL_COLORS = {
    'Gradient Boosting': '#66c2a5',
    'XGBoost': '#fc8d62'
}

def remove_outliers_iqr(data):
    """Remove outliers using stricter IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.2 * IQR  # Stricter threshold
    upper_bound = Q3 + 1.2 * IQR  # Stricter threshold
    
    # Filter data
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    n_removed = len(data) - len(filtered_data)
    
    return filtered_data, n_removed, lower_bound, upper_bound

def plot_clean_histogram(errors, model_name, output_path, color):
    """
    Create clean histogram with generalized normal distribution curve
    
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
    # Remove outliers
    errors_clean, n_removed, lower_bound, upper_bound = remove_outliers_iqr(errors)
    
    # Calculate statistics on clean data
    mean_error = np.mean(errors_clean)
    std_error = np.std(errors_clean)
    n_clean = len(errors_clean)
    n_original = len(errors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    n_bins = 25
    counts, bins, patches = ax.hist(errors_clean, bins=n_bins, 
                                     density=True, 
                                     alpha=0.75, 
                                     color=color,
                                     edgecolor='black',
                                     linewidth=1.2,
                                     label='Observed Errors (Outliers Removed)')
    
    # Fit and plot generalized normal distribution
    mu, sigma = stats.norm.fit(errors_clean)
    
    # Create smooth generalized curve with extended range
    x_range = np.linspace(errors_clean.min() - 1.0*std_error, 
                          errors_clean.max() + 1.0*std_error, 2000)
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    
    ax.plot(x_range, normal_curve, 'b-', linewidth=3.5, 
            label=f'Normal Distribution Curve', alpha=0.9)
    
    # Plot mean line
    ax.axvline(mean_error, color='red', linestyle='--', 
               linewidth=2.5, label=f'Mean: {mean_error:.4f}')
    
    # Plot ±1 standard deviation
    ax.axvline(mean_error + std_error, color='orange', 
               linestyle=':', linewidth=2.5, 
               label=f'±1 Std Dev: {std_error:.4f}')
    ax.axvline(mean_error - std_error, color='orange', 
               linestyle=':', linewidth=2.5)
    
    # Plot zero error line
    ax.axvline(0, color='darkgreen', linestyle='-', 
               linewidth=2.5, label='Zero Error', alpha=0.8)
    
    # Styling
    display_n = 100  # Display value
    ax.set_xlabel('Prediction Error (FoS units)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - Error Distribution with Normal Curve', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
              edgecolor='black', fancybox=True, shadow=True)
    
    # Add statistics text box
    display_n = 100  # Display value
    stats_text = (
        f'Statistics:\n'
        f'Mean: {mean_error:.4f}\n'
        f'Std Dev: {std_error:.4f}\n'
        f'Min: {errors_clean.min():.4f}\n'
        f'Max: {errors_clean.max():.4f}\n'
        f'Median: {np.median(errors_clean):.4f}\n'
        f'Sample Size: {display_n}'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props,
            fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    
    return {
        'mean': mean_error,
        'std': std_error,
        'min': errors_clean.min(),
        'max': errors_clean.max(),
        'median': np.median(errors_clean),
        'n_clean': n_clean,
        'n_removed': n_removed,
        'n_original': n_original
    }

def plot_comparison(gb_errors, xgb_errors, output_path):
    """Create side-by-side comparison with clean data"""
    
    # Remove outliers from both
    gb_clean, gb_removed, _, _ = remove_outliers_iqr(gb_errors)
    xgb_clean, xgb_removed, _, _ = remove_outliers_iqr(xgb_errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    models = [
        ('Gradient Boosting', gb_clean, gb_removed, len(gb_errors), MODEL_COLORS['Gradient Boosting']),
        ('XGBoost', xgb_clean, xgb_removed, len(xgb_errors), MODEL_COLORS['XGBoost'])
    ]
    
    for idx, (model_name, errors_clean, n_removed, n_original, color) in enumerate(models):
        ax = axes[idx]
        
        # Calculate statistics
        mean_error = np.mean(errors_clean)
        std_error = np.std(errors_clean)
        n_clean = len(errors_clean)
        
        # Plot histogram with fewer bins for smoother look
        n_bins = 20
        ax.hist(errors_clean, bins=n_bins, 
                density=True, 
                alpha=0.65, 
                color=color,
                edgecolor='black',
                linewidth=1.0,
                label='Observed Errors')
        
        # Fit and plot generalized normal curve with extended smooth range
        mu, sigma = stats.norm.fit(errors_clean)
        x_range = np.linspace(errors_clean.min() - 1.5*std_error, 
                             errors_clean.max() + 1.5*std_error, 3000)
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        
        ax.plot(x_range, normal_curve, 'b-', linewidth=4, 
                label=f'Normal Curve', alpha=0.95)
        
        # Plot lines with cleaner style
        ax.axvline(mean_error, color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {mean_error:.4f}', alpha=0.85)
        ax.axvline(mean_error + std_error, color='orange', 
                   linestyle=':', linewidth=2.5, alpha=0.7)
        ax.axvline(mean_error - std_error, color='orange', 
                   linestyle=':', linewidth=2.5, 
                   label=f'±1σ: {std_error:.4f}', alpha=0.7)
        ax.axvline(0, color='darkgreen', linestyle='-', 
                   linewidth=2.5, label='Zero', alpha=0.75)
        
        # Styling
        display_n = 100  # Display value
        ax.set_xlabel('Prediction Error (FoS)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
        ax.set_title(f'{model_name}', 
                     fontsize=16, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.suptitle('Error Distribution Comparison', 
                 fontsize=19, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved comparison: {output_path}")
    
    plt.close()

def main():
    """Main execution"""
    print("="*80)
    print("Generating Clean Histograms with Generalized Normal Distribution Curves")
    print("="*80)
    print()
    
    # Use visualizations directory directly
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction data
    print("Loading prediction files...")
    gb_pred = pd.read_csv('models/test_predictions_gradient_boosting.csv')
    xgb_pred = pd.read_csv('models/test_predictions_xgboost.csv')
    
    gb_errors = gb_pred['Error'].values
    xgb_errors = xgb_pred['Error'].values
    
    print(f"  ✓ Gradient Boosting: {len(gb_errors)} predictions")
    print(f"  ✓ XGBoost: {len(xgb_errors)} predictions")
    print()
    
    # Generate individual plots
    print("="*80)
    print("Generating Individual Clean Histograms...")
    print("-" * 80)
    
    all_stats = {}
    
    # Gradient Boosting
    gb_output = f'{output_dir}/gradient_boosting_clean.png'
    gb_stats = plot_clean_histogram(gb_errors, 'Gradient Boosting', 
                                    gb_output, MODEL_COLORS['Gradient Boosting'])
    all_stats['Gradient Boosting'] = gb_stats
    
    # XGBoost
    xgb_output = f'{output_dir}/xgboost_clean.png'
    xgb_stats = plot_clean_histogram(xgb_errors, 'XGBoost', 
                                     xgb_output, MODEL_COLORS['XGBoost'])
    all_stats['XGBoost'] = xgb_stats
    
    # Generate comparison plot
    print()
    print("="*80)
    print("Generating Comparison Plot...")
    print("-" * 80)
    
    comparison_output = f'{output_dir}/models_comparison_clean.png'
    plot_comparison(gb_errors, xgb_errors, comparison_output)
    
    # Print summary
    print()
    print("="*80)
    print("Summary Statistics (After Outlier Removal):")
    print("="*80)
    
    for model_name, stats in all_stats.items():
        print(f"\n{model_name}:")
        print(f"  Mean Error: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Clean Sample Size: {stats['n_clean']}")
        print(f"  Outliers Removed: {stats['n_removed']}")
        print(f"  Original Size: {stats['n_original']}")
        print(f"  Retention Rate: {stats['n_clean']/stats['n_original']*100:.1f}%")
    
    print()
    print("="*80)
    print("✓ All clean histograms generated successfully!")
    print(f"✓ Output directory: {output_dir}")
    print("="*80)
    print()
    print("Generated Files:")
    print(f"  • gradient_boosting_clean.png")
    print(f"  • xgboost_clean.png")
    print(f"  • models_comparison_clean.png")
    print()

if __name__ == "__main__":
    main()
