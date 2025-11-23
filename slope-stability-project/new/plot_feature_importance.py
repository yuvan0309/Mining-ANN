#!/usr/bin/env python3
"""
Feature Importance Visualization for Random Forest Model
Generates horizontal bar chart showing top features influencing FoS prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def plot_feature_importance(model_path='models/random_forest.pkl', 
                            output_dir='visualizations',
                            top_n=10):
    """
    Plot feature importance from trained Random Forest model.
    
    Parameters:
    -----------
    model_path : str
        Path to saved Random Forest model
    output_dir : str
        Directory to save the plot
    top_n : int
        Number of top features to display
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the trained Random Forest model
    try:
        rf_model = joblib.load(model_path)
        print(f"✓ Loaded Random Forest model from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train_models.py")
        return
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Feature names (based on your dataset structure)
    feature_names = [
        'Mean Cohesion',
        'Mean Friction Angle', 
        'Mean Unit Weight',
        'Mean Ru',
        'Laterite Cohesion',
        'Laterite Friction',
        'Laterite Unit Weight',
        'Laterite Ru',
        'Phyllitic Clay Cohesion',
        'Phyllitic Clay Friction',
        'Phyllitic Clay Unit Weight',
        'Phyllitic Clay Ru',
        'Lumpy Iron Ore Cohesion',
        'Lumpy Iron Ore Friction',
        'Lumpy Iron Ore Unit Weight',
        'Lumpy Iron Ore Ru',
        'Limonitic Clay Cohesion',
        'Limonitic Clay Friction',
        'Limonitic Clay Unit Weight',
        'Limonitic Clay Ru',
        'Manganiferous Clay Cohesion',
        'Manganiferous Clay Friction',
        'Manganiferous Clay Unit Weight',
        'Manganiferous Clay Ru',
        'Siliceous Clay Cohesion',
        'Siliceous Clay Friction',
        'Siliceous Clay Unit Weight',
        'Siliceous Clay Ru',
        'BHQ Cohesion',
        'BHQ Friction',
        'BHQ Unit Weight',
        'BHQ Ru',
        'Schist Unit Weight',
        'Schist Friction',
        'Schist Cohesion',
        'Schist Ru',
        'Mine Location',
        'Point Index',
        'Season'
    ]
    
    # Adjust feature names if model has fewer features
    if len(importances) == 4:
        # Model trained on averaged features only
        feature_names = [
            'Mean Cohesion',
            'Mean Friction Angle',
            'Mean Unit Weight',
            'Mean Ru'
        ]
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    })
    
    # Sort by importance and get top N
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    top_features = feature_importance_df.tail(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_features)), 
                   top_features['Importance'],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))),
                   edgecolor='black',
                   linewidth=1.2)
    
    # Customize the plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=11)
    ax.set_xlabel('Relative Importance', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - Random Forest Model\nFactor of Safety Prediction', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Importance'] + 0.002, i, f"{row['Importance']:.2%}", 
                va='center', fontsize=10, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set background color
    ax.set_facecolor('#f0f0f5')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'feature_importance_random_forest.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved feature importance plot: {output_path}")
    
    # Also save as PDF for publication quality
    output_path_pdf = output_dir / 'feature_importance_random_forest.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version: {output_path_pdf}")
    
    plt.close()
    
    # Print feature importance summary
    print("\n" + "="*60)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*60)
    for idx, row in top_features.iloc[::-1].iterrows():
        print(f"{row['Feature']:30s} : {row['Importance']:6.2%}")
    print("="*60)
    
    # Save to CSV
    csv_path = output_dir / 'feature_importance.csv'
    feature_importance_df.sort_values('Importance', ascending=False).to_csv(csv_path, index=False)
    print(f"✓ Saved feature importance data: {csv_path}")


def plot_all_model_importances(models_dir='models', output_dir='visualizations'):
    """
    Plot feature importance for all tree-based models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    
    # Models that have feature_importances_ attribute
    tree_models = {
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'XGBoost': 'xgboost.pkl',
        'LightGBM': 'lightgbm.pkl'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    feature_names = ['Cohesion', 'Friction Angle', 'Unit Weight', 'Ru']
    
    for idx, (model_name, model_file) in enumerate(tree_models.items()):
        model_path = models_dir / model_file
        
        try:
            model = joblib.load(model_path)
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                print(f"⚠ {model_name} does not have feature_importances_")
                continue
            
            # Create DataFrame
            df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Plot
            ax = axes[idx]
            bars = ax.barh(df['Feature'], df['Importance'],
                          color=plt.cm.viridis(np.linspace(0.3, 0.9, len(df))),
                          edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel('Relative Importance', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.set_facecolor('#f0f0f5')
            
            # Add value labels
            for i, (feat, imp) in enumerate(zip(df['Feature'], df['Importance'])):
                ax.text(imp + 0.01, i, f"{imp:.2%}", va='center', fontsize=9, fontweight='bold')
            
            print(f"✓ Plotted {model_name}")
            
        except FileNotFoundError:
            print(f"⚠ Model not found: {model_path}")
            axes[idx].text(0.5, 0.5, f'{model_name}\nNot Available', 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
    
    plt.suptitle('Feature Importance Comparison - All Tree-Based Models', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'feature_importance_all_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Change to script directory
    script_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE VISUALIZATION")
    print("="*60 + "\n")
    
    # Plot Random Forest feature importance (main plot)
    plot_feature_importance(
        model_path=script_dir / 'models' / 'random_forest.pkl',
        output_dir=script_dir / 'visualizations',
        top_n=10
    )
    
    print("\n" + "="*60)
    print("COMPARISON PLOT FOR ALL MODELS")
    print("="*60 + "\n")
    
    # Plot all models comparison
    plot_all_model_importances(
        models_dir=script_dir / 'models',
        output_dir=script_dir / 'visualizations'
    )
    
    print("\n✅ Feature importance visualization complete!")
    print(f"📊 Check the 'visualizations' folder for output files.\n")
