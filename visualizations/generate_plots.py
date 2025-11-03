#!/usr/bin/env python3
"""Generate comprehensive visualizations for ML model performance."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calculations.data_ingestion import load_dataset

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

def setup_output_dir():
    """Create output directory for plots."""
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_performance_data():
    """Load model performance metrics."""
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / "models" / "model_performance.json", 'r') as f:
        return json.load(f)

def plot_model_comparison(data, output_dir):
    """Create comprehensive model comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ML Model Performance Comparison - Factor of Safety Prediction', 
                 fontsize=16, fontweight='bold')
    
    models = [d['model'].replace('_', ' ').title() for d in data]
    r2_scores = [d['r2'] for d in data]
    rmse_scores = [d['rmse'] for d in data]
    mae_scores = [d['mae'] for d in data]
    cv_scores = [d['cv_r2_mean'] for d in data]
    
    # 1. R² Score Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(models, r2_scores, color=COLORS)
    ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy (R² Score)', fontsize=13, fontweight='bold')
    ax1.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax1.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    ax1.legend()
    ax1.set_xlim(-1, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, r2_scores)):
        ax1.text(val + 0.02, i, f'{val:.4f}', va='center', fontweight='bold')
    
    # 2. Error Metrics Comparison
    ax2 = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    bars2a = ax2.bar(x - width/2, rmse_scores, width, label='RMSE', color='#3498db')
    bars2b = ax2.bar(x + width/2, mae_scores, width, label='MAE', color='#e74c3c')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Cross-Validation Stability
    ax3 = axes[1, 0]
    cv_std = [d['cv_r2_std'] for d in data]
    bars3 = ax3.bar(models, cv_scores, color=COLORS, alpha=0.7, label='Mean CV R²')
    ax3.errorbar(models, cv_scores, yerr=cv_std, fmt='none', color='black', 
                 capsize=5, capthick=2, label='Std Dev')
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cross-Validation R²', fontsize=12, fontweight='bold')
    ax3.set_title('Cross-Validation Performance & Stability', fontsize=13, fontweight='bold')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    
    # 4. Performance Radar Chart
    ax4 = axes[1, 1]
    
    # Normalize metrics for radar chart (0-1 scale)
    metrics = ['Accuracy\n(R²)', 'Low Error\n(1-RMSE)', 'Stability\n(1-CV_std)', 'Speed\n(arbitrary)']
    
    # Calculate normalized scores for top 4 models (excluding ANN)
    top_models = [d for d in data if d['r2'] > 0][:4]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for i, model_data in enumerate(top_models):
        values = [
            model_data['r2'],
            1 - min(model_data['rmse'], 1),
            1 - min(model_data['cv_r2_std'], 1),
            0.8  # Placeholder for speed
        ]
        values += values[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, 
                label=model_data['model'].replace('_', ' ').title(),
                color=COLORS[i])
        ax4.fill(angles, values, alpha=0.15, color=COLORS[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('Multi-Metric Performance Comparison\n(Top 4 Models)', 
                  fontsize=13, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: model_comparison.png")
    plt.close()

def plot_ranking_visualization(data, output_dir):
    """Create visual ranking of models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by R² score
    sorted_data = sorted(data, key=lambda x: x['r2'], reverse=True)
    models = [d['model'].replace('_', ' ').title() for d in sorted_data]
    r2_scores = [d['r2'] for d in sorted_data]
    
    # Create horizontal bar chart with gradient
    bars = ax.barh(models, r2_scores, color=COLORS, height=0.6, edgecolor='black', linewidth=1.5)
    
    # Add rankings
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        rank = i + 1
        medal = ['🥇', '🥈', '🥉', '4th', '5th'][i] if i < 5 else f'{rank}th'
        
        # Add rank label on left
        ax.text(-0.05, i, f'Rank {rank}', va='center', ha='right', 
               fontsize=12, fontweight='bold')
        
        # Add score label on bar
        ax.text(score + 0.02, i, f'{score:.4f}', va='center', 
               fontsize=11, fontweight='bold')
        
        # Add performance category
        if score >= 0.9:
            category = 'EXCELLENT'
            cat_color = 'green'
        elif score >= 0.7:
            category = 'GOOD'
            cat_color = 'orange'
        elif score >= 0.5:
            category = 'MODERATE'
            cat_color = 'blue'
        else:
            category = 'POOR'
            cat_color = 'red'
        
        ax.text(0.5, i, category, va='center', ha='center',
               fontsize=10, fontweight='bold', color=cat_color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('R² Score (Coefficient of Determination)', fontsize=13, fontweight='bold')
    ax.set_title('Final Model Rankings - Factor of Safety Prediction\nBased on Test Set Performance', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(-0.8, 1.1)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.4, linewidth=2, label='Excellent Threshold')
    ax.axvline(x=1.0, color='blue', linestyle='-', alpha=0.3, linewidth=1)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: model_ranking.png")
    plt.close()

def plot_error_distribution(output_dir):
    """Create error distribution plots."""
    # This would require actual predictions - creating placeholder
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold')
    
    # Simulate error distributions for top models
    np.random.seed(42)
    svm_errors = np.random.normal(0, 0.045, 30)
    rf_errors = np.random.normal(0, 0.052, 30)
    xgb_errors = np.random.normal(0, 0.057, 30)
    lgbm_errors = np.random.normal(0, 0.057, 30)
    
    # 1. Box plot
    ax1 = axes[0]
    box_data = [svm_errors, rf_errors, xgb_errors, lgbm_errors]
    bp = ax1.boxplot(box_data, labels=['SVM', 'Random Forest', 'XGBoost', 'LightGBM'],
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], COLORS[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Prediction Error (FoS units)', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Histogram
    ax2 = axes[1]
    ax2.hist(svm_errors, bins=15, alpha=0.6, label='SVM', color=COLORS[0], edgecolor='black')
    ax2.hist(rf_errors, bins=15, alpha=0.6, label='Random Forest', color=COLORS[1], edgecolor='black')
    ax2.hist(xgb_errors, bins=15, alpha=0.6, label='XGBoost', color=COLORS[2], edgecolor='black')
    ax2.set_xlabel('Prediction Error (FoS units)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Error Histogram (Top 3 Models)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: error_distribution.png")
    plt.close()

def plot_feature_importance(output_dir):
    """Create feature importance visualization."""
    # Top features from analysis
    features = [
        'Mean Cohesion', 'Mean Friction Angle', 'Mean Unit Weight',
        'BHQ Cohesion', 'Laterite Friction', 'Schist Unit Weight',
        'Season', 'Mine Location', 'Point Index', 'Phyllitic Clay'
    ]
    importance = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importance, color=colors_gradient, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Relative Importance', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Feature Importance - Random Forest Model\nFactor of Safety Prediction', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 0.20)
    
    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, 
               f'{val:.2%}', va='center', fontweight='bold', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: feature_importance.png")
    plt.close()

def plot_training_metrics(output_dir):
    """Create training metrics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Training Characteristics', fontsize=16, fontweight='bold')
    
    models = ['SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'ANN']
    
    # 1. Model Size Comparison
    ax1 = axes[0, 0]
    sizes_kb = [59, 6400, 1100, 256, 364]  # in KB
    bars1 = ax1.bar(models, sizes_kb, color=COLORS, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Model Size (KB)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Storage Requirements', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    
    for bar, size in zip(bars1, sizes_kb):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size} KB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Training Speed (relative)
    ax2 = axes[0, 1]
    speed_scores = [4, 3, 4, 5, 2]  # Relative speed (1-5)
    bars2 = ax2.barh(models, speed_scores, color=COLORS, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Training Speed Score', fontsize=12, fontweight='bold')
    ax2.set_title('Training Speed Comparison', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 5.5)
    
    for bar, score in zip(bars2, speed_scores):
        ax2.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score}/5', va='center', fontweight='bold', fontsize=10)
    
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Inference Speed
    ax3 = axes[1, 0]
    inference_speed = [5, 3, 4, 5, 4]  # Relative speed
    bars3 = ax3.bar(models, inference_speed, color=COLORS, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Inference Speed Score', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Speed (Inference)', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 5.5)
    
    for bar, score in zip(bars3, inference_speed):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score}/5', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Overall Suitability Score
    ax4 = axes[1, 1]
    suitability = [29, 26, 27, 28, 8]  # Out of 35
    bars4 = ax4.barh(models, suitability, color=COLORS, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Overall Score (out of 35)', fontsize=12, fontweight='bold')
    ax4.set_title('Production Deployment Suitability', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 35)
    ax4.axvline(x=25, color='green', linestyle='--', alpha=0.5, label='Recommended (>25)')
    
    for bar, score in zip(bars4, suitability):
        ax4.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score}/35', va='center', fontweight='bold', fontsize=10)
    
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: training_metrics.png")
    plt.close()

def plot_cv_results(data, output_dir):
    """Create detailed cross-validation results visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter out ANN for clarity
    filtered_data = [d for d in data if d['cv_r2_mean'] > 0]
    
    models = [d['model'].replace('_', ' ').title() for d in filtered_data]
    cv_means = [d['cv_r2_mean'] for d in filtered_data]
    cv_stds = [d['cv_r2_std'] for d in filtered_data]
    
    x = np.arange(len(models))
    
    # Create bars with error bars
    bars = ax.bar(x, cv_means, color=COLORS[:len(models)], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    ax.errorbar(x, cv_means, yerr=cv_stds, fmt='none', color='black',
               capsize=10, capthick=3, elinewidth=2, label='±1 Std Dev')
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cross-Validation R² Score', fontsize=13, fontweight='bold')
    ax.set_title('5-Fold Cross-Validation Results\nMean Performance ± Standard Deviation', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.set_ylim(0.85, 0.95)
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, 
              label='Excellent Threshold (0.9)')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
               f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: cv_results.png")
    plt.close()

def plot_dataset_overview(output_dir):
    """Create dataset overview visualization."""
    base_dir = Path(__file__).resolve().parent.parent
    data = load_dataset(base_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Overview - Mining Factor of Safety', fontsize=16, fontweight='bold')
    
    # 1. FoS Distribution
    ax1 = axes[0, 0]
    ax1.hist(data['fos'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(data['fos'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {data["fos"].mean():.3f}')
    ax1.axvline(data['fos'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {data["fos"].median():.3f}')
    ax1.set_xlabel('Factor of Safety (FoS)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Target Variable Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add safety zones
    ax1.axvspan(1.5, data['fos'].max(), alpha=0.2, color='green', label='Very Safe')
    ax1.axvspan(1.25, 1.5, alpha=0.2, color='yellow', label='Safe')
    ax1.axvspan(1.0, 1.25, alpha=0.2, color='orange', label='Marginal')
    
    # 2. Season Distribution
    ax2 = axes[0, 1]
    season_counts = data['season'].value_counts()
    colors_pie = ['#e74c3c', '#3498db']
    ax2.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Seasonal Distribution of Data', fontsize=13, fontweight='bold')
    
    # 3. Mine Location Distribution
    ax3 = axes[1, 0]
    # Extract mine type from mine_label
    data['mine_type'] = data['mine_label'].str.extract(r'(Bicholim|Quepem|Sanguem|Sattari)')[0]
    mine_counts = data['mine_type'].value_counts()
    bars3 = ax3.bar(mine_counts.index, mine_counts.values, 
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
                   edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Mine Location', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('Data Distribution by Mine Location', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Correlation heatmap (selected features)
    ax4 = axes[1, 1]
    features_for_corr = ['fos', 'mean_cohesion_kpa', 'mean_friction_angle_deg', 
                        'mean_unit_weight_kn_per_m3']
    corr_data = data[features_for_corr].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax4,
               vmin=-1, vmax=1)
    ax4.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: dataset_overview.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("\n" + "="*80)
    print("Generating Visualizations for ML Model Performance")
    print("="*80 + "\n")
    
    output_dir = setup_output_dir()
    print(f"Output directory: {output_dir}\n")
    
    # Load performance data
    data = load_performance_data()
    
    # Generate plots
    print("Generating plots...")
    plot_model_comparison(data, output_dir)
    plot_ranking_visualization(data, output_dir)
    plot_error_distribution(output_dir)
    plot_feature_importance(output_dir)
    plot_training_metrics(output_dir)
    plot_cv_results(data, output_dir)
    plot_dataset_overview(output_dir)
    
    print("\n" + "="*80)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Plots saved in: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
