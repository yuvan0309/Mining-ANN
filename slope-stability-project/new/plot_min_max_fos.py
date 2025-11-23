#!/usr/bin/env python3
"""
Generate Min/Max FoS Plot for Models
Shows the range of FoS predictions for Gradient Boosting and XGBoost models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
MODEL_DIR = Path(__file__).parent / 'models'
OUTPUT_DIR = Path(__file__).parent / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load test predictions
gb_predictions = pd.read_csv(MODEL_DIR / 'test_predictions_gradient_boosting.csv')
xgb_predictions = pd.read_csv(MODEL_DIR / 'test_predictions_xgboost.csv')

# Calculate min and max for each model
gb_min = gb_predictions['Predicted FoS'].min()
gb_max = gb_predictions['Predicted FoS'].max()
gb_mean = gb_predictions['Predicted FoS'].mean()
gb_median = gb_predictions['Predicted FoS'].median()

xgb_min = xgb_predictions['Predicted FoS'].min()
xgb_max = xgb_predictions['Predicted FoS'].max()
xgb_mean = xgb_predictions['Predicted FoS'].mean()
xgb_median = xgb_predictions['Predicted FoS'].median()

# Actual FoS min/max (for comparison)
actual_min = gb_predictions['Actual FoS'].min()
actual_max = gb_predictions['Actual FoS'].max()
actual_mean = gb_predictions['Actual FoS'].mean()
actual_median = gb_predictions['Actual FoS'].median()

print("="*60)
print("MIN/MAX FoS VALUES - MODEL COMPARISON")
print("="*60)
print(f"\nActual FoS:")
print(f"  Min:    {actual_min:.4f}")
print(f"  Max:    {actual_max:.4f}")
print(f"  Mean:   {actual_mean:.4f}")
print(f"  Median: {actual_median:.4f}")
print(f"  Range:  {actual_max - actual_min:.4f}")

print(f"\nGradient Boosting Predictions:")
print(f"  Min:    {gb_min:.4f}")
print(f"  Max:    {gb_max:.4f}")
print(f"  Mean:   {gb_mean:.4f}")
print(f"  Median: {gb_median:.4f}")
print(f"  Range:  {gb_max - gb_min:.4f}")

print(f"\nXGBoost Predictions:")
print(f"  Min:    {xgb_min:.4f}")
print(f"  Max:    {xgb_max:.4f}")
print(f"  Mean:   {xgb_mean:.4f}")
print(f"  Median: {xgb_median:.4f}")
print(f"  Range:  {xgb_max - xgb_min:.4f}")

print("\n" + "="*60)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Min/Max Comparison Bar Chart
models = ['Actual', 'Gradient\nBoosting', 'XGBoost']
min_values = [actual_min, gb_min, xgb_min]
max_values = [actual_max, gb_max, xgb_max]
mean_values = [actual_mean, gb_mean, xgb_mean]

x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, min_values, width, label='Minimum FoS', 
                color='#ef4444', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, max_values, width, label='Maximum FoS', 
                color='#10b981', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax1.bar(x + width, mean_values, width, label='Mean FoS', 
                color='#3b82f6', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('Factor of Safety (FoS)', fontsize=14, fontweight='bold')
ax1.set_title('Min/Max/Mean FoS Comparison - Test Set (73 samples)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(max_values) * 1.15)

# Add safety zones
ax1.axhspan(0, 1.0, alpha=0.1, color='red', label='Critical (<1.0)')
ax1.axhspan(1.0, 1.3, alpha=0.1, color='orange', label='Warning (1.0-1.3)')
ax1.axhspan(1.3, 1.5, alpha=0.1, color='yellow', label='Caution (1.3-1.5)')
ax1.axhspan(1.5, ax1.get_ylim()[1], alpha=0.1, color='green', label='Safe (≥1.5)')

# Plot 2: Range Visualization with Error Bars
fig2_data = {
    'Model': models,
    'Min': min_values,
    'Max': max_values,
    'Mean': mean_values,
    'Range': [max_values[i] - min_values[i] for i in range(len(models))]
}

# Create range plot
for i, model in enumerate(models):
    # Draw vertical line for range
    ax2.plot([i, i], [min_values[i], max_values[i]], 
            color='#64748b', linewidth=3, zorder=1)
    
    # Draw min marker
    ax2.scatter(i, min_values[i], s=200, color='#ef4444', 
               marker='v', zorder=3, edgecolor='black', linewidth=2, 
               label='Minimum' if i == 0 else '')
    
    # Draw max marker
    ax2.scatter(i, max_values[i], s=200, color='#10b981', 
               marker='^', zorder=3, edgecolor='black', linewidth=2,
               label='Maximum' if i == 0 else '')
    
    # Draw mean marker
    ax2.scatter(i, mean_values[i], s=250, color='#3b82f6', 
               marker='D', zorder=4, edgecolor='black', linewidth=2,
               label='Mean' if i == 0 else '')
    
    # Add range text
    range_val = max_values[i] - min_values[i]
    ax2.text(i + 0.15, (min_values[i] + max_values[i]) / 2, 
            f'Range:\n{range_val:.3f}',
            fontsize=10, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

ax2.set_xlabel('Model', fontsize=14, fontweight='bold')
ax2.set_ylabel('Factor of Safety (FoS)', fontsize=14, fontweight='bold')
ax2.set_title('FoS Prediction Range by Model', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, fontsize=12)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_xlim(-0.5, len(models) - 0.5)
ax2.set_ylim(0, max(max_values) * 1.15)

# Add safety zones to second plot
ax2.axhspan(0, 1.0, alpha=0.1, color='red')
ax2.axhspan(1.0, 1.3, alpha=0.1, color='orange')
ax2.axhspan(1.3, 1.5, alpha=0.1, color='yellow')
ax2.axhspan(1.5, ax2.get_ylim()[1], alpha=0.1, color='green')

# Add horizontal reference lines
for val, label, color in [(1.0, 'FoS=1.0', 'red'), 
                           (1.3, 'FoS=1.3', 'orange'),
                           (1.5, 'FoS=1.5', 'green')]:
    ax2.axhline(y=val, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.text(len(models) - 0.5, val + 0.03, label, 
            fontsize=9, color=color, fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'min_max_fos_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved: {OUTPUT_DIR / 'min_max_fos_comparison.png'}")

# Create additional detailed statistics plot
fig3, ax3 = plt.subplots(figsize=(14, 8))

# Box plot data
data_to_plot = [
    gb_predictions['Actual FoS'].values,
    gb_predictions['Predicted FoS'].values,
    xgb_predictions['Predicted FoS'].values
]

bp = ax3.boxplot(data_to_plot, labels=models, patch_artist=True,
                 showmeans=True, meanline=False,
                 widths=0.6,
                 boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                 whiskerprops=dict(linewidth=2),
                 capprops=dict(linewidth=2),
                 medianprops=dict(color='red', linewidth=3),
                 meanprops=dict(marker='D', markerfacecolor='yellow', 
                               markeredgecolor='black', markersize=10))

# Color boxes differently
colors = ['#94a3b8', '#22c55e', '#3b82f6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax3.set_xlabel('Model', fontsize=14, fontweight='bold')
ax3.set_ylabel('Factor of Safety (FoS)', fontsize=14, fontweight='bold')
ax3.set_title('FoS Distribution - Box Plot Analysis', 
              fontsize=16, fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add safety zones
ax3.axhspan(0, 1.0, alpha=0.1, color='red')
ax3.axhspan(1.0, 1.3, alpha=0.1, color='orange')
ax3.axhspan(1.3, 1.5, alpha=0.1, color='yellow')
ax3.axhspan(1.5, ax3.get_ylim()[1], alpha=0.1, color='green')

# Add reference lines
for val, label in [(1.0, 'Critical'), (1.3, 'Warning'), (1.5, 'Safe')]:
    ax3.axhline(y=val, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(0.5, val + 0.02, label, fontsize=9, alpha=0.7)

# Add legend for box plot elements
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', 
           markeredgecolor='black', markersize=10, label='Mean'),
    Line2D([0], [0], color='black', linewidth=2, label='Q1/Q3 (Box)'),
    Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Whiskers (1.5×IQR)')
]
ax3.legend(handles=legend_elements, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fos_box_plot_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Box plot saved: {OUTPUT_DIR / 'fos_box_plot_comparison.png'}")

# Create summary table
print("\n" + "="*60)
print("DETAILED STATISTICS SUMMARY")
print("="*60)

summary_data = {
    'Metric': ['Minimum', 'Maximum', 'Range', 'Mean', 'Median', 'Std Dev', 
               'Q1 (25%)', 'Q3 (75%)', 'IQR'],
    'Actual': [
        f"{actual_min:.4f}",
        f"{actual_max:.4f}",
        f"{actual_max - actual_min:.4f}",
        f"{actual_mean:.4f}",
        f"{actual_median:.4f}",
        f"{gb_predictions['Actual FoS'].std():.4f}",
        f"{gb_predictions['Actual FoS'].quantile(0.25):.4f}",
        f"{gb_predictions['Actual FoS'].quantile(0.75):.4f}",
        f"{gb_predictions['Actual FoS'].quantile(0.75) - gb_predictions['Actual FoS'].quantile(0.25):.4f}"
    ],
    'Gradient Boosting': [
        f"{gb_min:.4f}",
        f"{gb_max:.4f}",
        f"{gb_max - gb_min:.4f}",
        f"{gb_mean:.4f}",
        f"{gb_median:.4f}",
        f"{gb_predictions['Predicted FoS'].std():.4f}",
        f"{gb_predictions['Predicted FoS'].quantile(0.25):.4f}",
        f"{gb_predictions['Predicted FoS'].quantile(0.75):.4f}",
        f"{gb_predictions['Predicted FoS'].quantile(0.75) - gb_predictions['Predicted FoS'].quantile(0.25):.4f}"
    ],
    'XGBoost': [
        f"{xgb_min:.4f}",
        f"{xgb_max:.4f}",
        f"{xgb_max - xgb_min:.4f}",
        f"{xgb_mean:.4f}",
        f"{xgb_median:.4f}",
        f"{xgb_predictions['Predicted FoS'].std():.4f}",
        f"{xgb_predictions['Predicted FoS'].quantile(0.25):.4f}",
        f"{xgb_predictions['Predicted FoS'].quantile(0.75):.4f}",
        f"{xgb_predictions['Predicted FoS'].quantile(0.75) - xgb_predictions['Predicted FoS'].quantile(0.25):.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv(MODEL_DIR / 'fos_min_max_summary.csv', index=False)
print(f"\n✓ Summary CSV saved: {MODEL_DIR / 'fos_min_max_summary.csv'}")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nFiles created:")
print(f"  1. {OUTPUT_DIR / 'min_max_fos_comparison.png'}")
print(f"  2. {OUTPUT_DIR / 'fos_box_plot_comparison.png'}")
print(f"  3. {MODEL_DIR / 'fos_min_max_summary.csv'}")
print("\n")
