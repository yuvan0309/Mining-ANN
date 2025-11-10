#!/usr/bin/env python3
"""
Generate Architecture and Flow Diagrams for FoS Prediction System
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_architecture_diagram():
    """Create system architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'FACTOR OF SAFETY (FoS) PREDICTION SYSTEM', 
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(8, 11, 'Architecture Diagram with Ru (Pore Pressure Ratio) Integration',
            ha='center', fontsize=12, style='italic', color='#34495e')
    
    # Color scheme
    data_color = '#3498db'  # Blue
    process_color = '#2ecc71'  # Green
    model_color = '#e74c3c'  # Red
    output_color = '#f39c12'  # Orange
    storage_color = '#9b59b6'  # Purple
    
    # ===== LAYER 1: DATA INPUT =====
    y_start = 9.5
    
    # Input Data Sources
    ax.add_patch(FancyBboxPatch((0.5, y_start), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=data_color, edgecolor='black', linewidth=2))
    ax.text(2, y_start+0.5, 'Pre-Monsoon Data\n(180 samples)', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((4, y_start), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=data_color, edgecolor='black', linewidth=2))
    ax.text(5.5, y_start+0.5, 'Post-Monsoon Data\n(181 samples)', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((7.5, y_start), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=data_color, edgecolor='black', linewidth=2))
    ax.text(9, y_start+0.5, 'Ru Values\n(Pore Pressure)', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.text(11.5, y_start+0.5, 'Total: 361 samples\n10 mines (A, B, C)', 
            ha='left', va='center', fontsize=8, style='italic', color='#7f8c8d')
    
    # Arrow down
    ax.arrow(5.5, y_start, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # ===== LAYER 2: DATA INGESTION =====
    y_start = 8
    
    ax.add_patch(FancyBboxPatch((2, y_start), 8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=process_color, edgecolor='black', linewidth=2))
    ax.text(6, y_start+0.4, 'DATA INGESTION MODULE (data_ingestion.py)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Sub-components
    ax.text(3, y_start-0.3, '• Parse CSV\n• Extract features\n• Handle Ru', 
            ha='left', va='top', fontsize=7, color='#2c3e50')
    ax.text(6.5, y_start-0.3, 'Features:\n• Cohesion (kPa)\n• Friction Angle (°)\n• Unit Weight (kN/m³)\n• Ru (optional)', 
            ha='left', va='top', fontsize=7, color='#2c3e50')
    
    # Arrow down
    ax.arrow(6, y_start-0.8, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # ===== LAYER 3: DATA SPLIT =====
    y_start = 6
    
    ax.add_patch(FancyBboxPatch((2, y_start), 8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=process_color, edgecolor='black', linewidth=2))
    ax.text(6, y_start+0.4, 'TRAIN-TEST SPLIT', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Split branches
    ax.arrow(6, y_start, -2, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', linestyle='--')
    ax.arrow(6, y_start, 2, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', linestyle='--')
    
    # Training data box
    ax.add_patch(FancyBboxPatch((2, y_start-1.8), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#16a085', edgecolor='black', linewidth=2))
    ax.text(3.5, y_start-1.4, '80% Training\n288 samples', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Testing data box
    ax.add_patch(FancyBboxPatch((7, y_start-1.8), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#e67e22', edgecolor='black', linewidth=2))
    ax.text(8.5, y_start-1.4, '20% Testing\n73 samples', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # ===== LAYER 4: MODEL TRAINING =====
    y_start = 3.5
    
    ax.add_patch(FancyBboxPatch((1, y_start), 9, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=model_color, edgecolor='black', linewidth=2, alpha=0.9))
    ax.text(5.5, y_start+1.1, 'MODEL TRAINING PHASE (train_models.py)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 6 Models
    models = ['SVM', 'Random\nForest', 'XGBoost', 'LightGBM', 'Gradient\nBoosting', 'ANN']
    x_positions = np.linspace(1.5, 9.5, 6)
    
    for i, (model, x) in enumerate(zip(models, x_positions)):
        color = '#FFD700' if model in ['XGBoost', 'Gradient\nBoosting'] else '#e74c3c'
        ax.add_patch(FancyBboxPatch((x-0.5, y_start+0.1), 1, 0.7, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(x, y_start+0.45, model, ha='center', va='center', 
                fontsize=6.5, fontweight='bold', color='white')
    
    ax.text(5.5, y_start-0.3, 'All 6 models trained on 80% data with StandardScaler', 
            ha='center', va='center', fontsize=8, style='italic', color='#2c3e50')
    
    # ===== LAYER 5: MODEL SELECTION =====
    y_start = 2
    
    ax.add_patch(FancyBboxPatch((2, y_start), 8, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#8e44ad', edgecolor='black', linewidth=2))
    ax.text(6, y_start+0.3, 'SELECT TOP 2 MODELS (Best R² on Training)', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Arrow down
    ax.arrow(6, y_start, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # ===== LAYER 6: TESTING =====
    y_start = 1
    
    ax.add_patch(FancyBboxPatch((2, y_start), 3.5, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFD700', edgecolor='black', linewidth=2))
    ax.text(3.75, y_start+0.25, 'Gradient Boosting\nR²=0.9426 ✅', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='#2c3e50')
    
    ax.add_patch(FancyBboxPatch((6.5, y_start), 3.5, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFD700', edgecolor='black', linewidth=2))
    ax.text(8.25, y_start+0.25, 'XGBoost\nR²=0.9420 ✅', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='#2c3e50')
    
    ax.text(6, y_start-0.3, 'Tested on 20% data (73 samples)', 
            ha='center', va='center', fontsize=8, style='italic', color='#7f8c8d')
    
    # ===== LAYER 7: OUTPUTS =====
    y_start = 0.2
    
    # Output boxes
    ax.add_patch(FancyBboxPatch((0.5, y_start-0.5), 2, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=output_color, edgecolor='black', linewidth=1.5))
    ax.text(1.5, y_start-0.3, 'Models\n(.pkl files)', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((3, y_start-0.5), 2, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=output_color, edgecolor='black', linewidth=1.5))
    ax.text(4, y_start-0.3, 'CSV Results\n(All metrics)', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((5.5, y_start-0.5), 2, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=output_color, edgecolor='black', linewidth=1.5))
    ax.text(6.5, y_start-0.3, 'Excel File\n(4 sheets)', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((8, y_start-0.5), 2, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=output_color, edgecolor='black', linewidth=1.5))
    ax.text(9, y_start-0.3, 'Visualizations\n(8 PNG files)', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((10.5, y_start-0.5), 2, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=output_color, edgecolor='black', linewidth=1.5))
    ax.text(11.5, y_start-0.3, 'JSON Metadata\n(Results)', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Right side annotations
    ax.text(14, 8, 'KEY FEATURES:', ha='left', fontsize=9, fontweight='bold', color='#2c3e50')
    ax.text(14, 7.6, '✓ Bishop\'s Method', ha='left', fontsize=7, color='#34495e')
    ax.text(14, 7.3, '✓ Ru Integration', ha='left', fontsize=7, color='#34495e')
    ax.text(14, 7.0, '✓ 6 ML Models', ha='left', fontsize=7, color='#34495e')
    ax.text(14, 6.7, '✓ Fine-tuned Top 2', ha='left', fontsize=7, color='#34495e')
    ax.text(14, 6.4, '✓ 80/20 Split', ha='left', fontsize=7, color='#34495e')
    ax.text(14, 6.1, '✓ StandardScaler', ha='left', fontsize=7, color='#34495e')
    
    ax.text(14, 5.5, 'PERFORMANCE:', ha='left', fontsize=9, fontweight='bold', color='#2c3e50')
    ax.text(14, 5.1, 'GB: R²=0.9426', ha='left', fontsize=7, color='#27ae60')
    ax.text(14, 4.8, 'XGB: R²=0.9420', ha='left', fontsize=7, color='#27ae60')
    ax.text(14, 4.5, 'Both ~94.2%!', ha='left', fontsize=7, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/inanotherlife/Mining ANN/new/visualizations/ARCHITECTURE_DIAGRAM.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Architecture diagram saved!")
    plt.close()


def create_flow_diagram():
    """Create detailed process flow diagram"""
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Title
    ax.text(7, 17.5, 'FoS PREDICTION SYSTEM - PROCESS FLOW', 
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(7, 17, 'Detailed Workflow from Data to Predictions',
            ha='center', fontsize=11, style='italic', color='#34495e')
    
    y = 16
    
    # ===== STEP 1: START =====
    ax.add_patch(Circle((7, y), 0.3, facecolor='#2ecc71', edgecolor='black', linewidth=2))
    ax.text(7, y, 'START', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.arrow(7, y-0.3, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 2: LOAD DATA =====
    ax.add_patch(FancyBboxPatch((4, y-0.4), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#3498db', edgecolor='black', linewidth=2))
    ax.text(7, y, 'STEP 1: Load Data\n(data_ingestion.py)', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.text(11, y, 'load_and_prepare_data()\nInclude Ru: Yes', 
            ha='left', va='center', fontsize=7, style='italic', color='#7f8c8d')
    
    ax.arrow(7, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.4
    
    # ===== STEP 3: PARSE CSV =====
    ax.add_patch(FancyBboxPatch((4, y-0.4), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#3498db', edgecolor='black', linewidth=2))
    ax.text(7, y, 'STEP 2: Parse CSV Structure\nExtract Features', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.text(2.5, y, 'Input:\nOverall Data.csv\n(361 samples)', 
            ha='right', va='center', fontsize=7, color='#2c3e50')
    
    ax.arrow(7, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.4
    
    # ===== STEP 4: FEATURE EXTRACTION =====
    ax.add_patch(FancyBboxPatch((3.5, y-0.6), 7, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#16a085', edgecolor='black', linewidth=2))
    ax.text(7, y+0.3, 'STEP 3: Feature Matrix (X)', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(7, y-0.1, '• Cohesion (c) - kPa', ha='center', fontsize=7, color='white')
    ax.text(7, y-0.25, '• Friction Angle (φ) - degrees', ha='center', fontsize=7, color='white')
    ax.text(7, y-0.4, '• Unit Weight (γ) - kN/m³', ha='center', fontsize=7, color='white')
    ax.text(7, y-0.55, '• Ru (Pore Pressure Ratio)', ha='center', fontsize=7, color='white')
    
    ax.arrow(7, y-0.7, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.6
    
    # ===== STEP 5: TRAIN-TEST SPLIT =====
    ax.add_patch(FancyBboxPatch((4, y-0.4), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#9b59b6', edgecolor='black', linewidth=2))
    ax.text(7, y, 'STEP 4: Train-Test Split\n80% / 20%', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.text(11.5, y, 'train_test_split()\nrandom_state=42', 
            ha='left', va='center', fontsize=7, style='italic', color='#7f8c8d')
    
    # Branch to training and testing
    ax.arrow(7, y-0.5, -2, -0.7, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    ax.arrow(7, y-0.5, 2, -0.7, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    y -= 1.6
    
    # Training branch
    ax.add_patch(FancyBboxPatch((1.5, y-0.3), 3, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#27ae60', edgecolor='black', linewidth=1.5))
    ax.text(3, y, '80% Training\n288 samples', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Testing branch
    ax.add_patch(FancyBboxPatch((9.5, y-0.3), 3, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#e67e22', edgecolor='black', linewidth=1.5))
    ax.text(11, y, '20% Testing\n73 samples\n(HELD OUT)', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.arrow(3, y-0.4, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 6: SCALING =====
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#2980b9', edgecolor='black', linewidth=2))
    ax.text(3, y, 'STEP 5: StandardScaler\nFit on Training', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.arrow(3, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 7: MODEL TRAINING =====
    ax.add_patch(FancyBboxPatch((1, y-1), 4, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#e74c3c', edgecolor='black', linewidth=2.5, alpha=0.9))
    ax.text(3, y+0.6, 'STEP 6: Train All 6 Models', ha='center', fontsize=9, fontweight='bold', color='white')
    
    models_left = ['1. SVM (RBF)', '2. Random Forest', '3. XGBoost (Fine-tuned)']
    models_right = ['4. LightGBM', '5. Gradient Boosting (Fine-tuned)', '6. ANN (MLP)']
    
    for i, model in enumerate(models_left):
        ax.text(1.5, y+0.2-i*0.25, model, ha='left', fontsize=7, color='white')
    for i, model in enumerate(models_right):
        ax.text(3, y+0.2-i*0.25, model, ha='left', fontsize=7, color='white')
    
    ax.text(3, y-0.8, 'Each model predicts on training set', ha='center', fontsize=7, style='italic', color='white')
    
    ax.arrow(3, y-1.1, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 2.0
    
    # ===== STEP 8: EVALUATE TRAINING =====
    ax.add_patch(FancyBboxPatch((1.5, y-0.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#8e44ad', edgecolor='black', linewidth=2))
    ax.text(3, y+0.2, 'STEP 7: Evaluate on Training', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(3, y-0.05, 'Calculate R², RMSE, MAE', ha='center', fontsize=7, color='white')
    ax.text(3, y-0.25, 'for each model', ha='center', fontsize=7, color='white')
    
    ax.arrow(3, y-0.6, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.4
    
    # ===== STEP 9: SELECT BEST 2 =====
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(3, y, 'STEP 8: Select Top 2\nby Training R²', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.arrow(3, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 10: SCALE TEST DATA =====
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#2980b9', edgecolor='black', linewidth=2))
    ax.text(3, y, 'STEP 9: Scale Test Data\n(Using Training Scaler)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Connect test data
    ax.arrow(11, y+3.5, 0, -3, head_width=0.15, head_length=0.1, 
             fc='#e67e22', ec='#e67e22', linewidth=2, linestyle='--')
    ax.arrow(11, y+0.5, -7.5, 0, head_width=0.15, head_length=0.1, 
             fc='#e67e22', ec='#e67e22', linewidth=2, linestyle='--')
    
    ax.arrow(3, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 11: TEST BEST 2 =====
    ax.add_patch(FancyBboxPatch((1, y-0.6), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFD700', edgecolor='black', linewidth=2.5))
    ax.text(3, y+0.3, 'STEP 10: Test Top 2 Models', ha='center', fontsize=9, fontweight='bold', color='#2c3e50')
    ax.text(3, y, '🥇 Gradient Boosting: R²=0.9426', ha='center', fontsize=7, color='#2c3e50')
    ax.text(3, y-0.25, '🥈 XGBoost: R²=0.9420', ha='center', fontsize=7, color='#2c3e50')
    ax.text(3, y-0.45, '73 test samples', ha='center', fontsize=6, style='italic', color='#7f8c8d')
    
    ax.arrow(3, y-0.7, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.5
    
    # ===== STEP 12: SAVE MODELS =====
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#16a085', edgecolor='black', linewidth=2))
    ax.text(3, y, 'STEP 11: Save Models\n(.pkl files)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.arrow(3, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.2
    
    # ===== STEP 13: GENERATE OUTPUTS =====
    ax.add_patch(FancyBboxPatch((0.5, y-0.8), 5, 1.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#34495e', edgecolor='black', linewidth=2.5))
    ax.text(3, y+0.4, 'STEP 12: Generate Outputs', ha='center', fontsize=9, fontweight='bold', color='white')
    
    outputs = [
        '✓ training_results.csv (6 models)',
        '✓ test_results.csv (2 models)',
        '✓ test_predictions_*.csv (73 rows each)',
        '✓ all_results.xlsx (4 sheets)',
        '✓ 8 PNG visualizations',
        '✓ JSON metadata'
    ]
    
    for i, output in enumerate(outputs):
        ax.text(3, y+0.1-i*0.18, output, ha='center', fontsize=6.5, color='white')
    
    ax.arrow(3, y-0.9, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)
    y -= 1.6
    
    # ===== STEP 14: END =====
    ax.add_patch(Circle((3, y), 0.3, facecolor='#e74c3c', edgecolor='black', linewidth=2))
    ax.text(3, y, 'END', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Right side: Key metrics
    ax.text(8, 14, 'KEY METRICS', ha='left', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.add_patch(FancyBboxPatch((7.5, 10), 6, 3.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    
    metrics_text = """
    TRAINING (80% - 288 samples):
    • Gradient Boosting: R²=0.9954
    • Random Forest: R²=0.9924
    • XGBoost: R²=0.9581
    
    TESTING (20% - 73 samples):
    • Gradient Boosting: R²=0.9426
    • XGBoost: R²=0.9420
    
    Overfitting Control:
    • XGBoost gap: 1.61% ✅
    • GB gap: 5.28% ✅
    """
    
    ax.text(8, 13.3, metrics_text, ha='left', va='top', fontsize=7, 
            color='#2c3e50', family='monospace')
    
    # Right side: File structure
    ax.text(8, 9, 'FILE STRUCTURE', ha='left', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.add_patch(FancyBboxPatch((7.5, 4.5), 6, 4, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    
    files_text = """
    new/
    ├── main_pipeline.py (Entry)
    ├── data_ingestion.py
    ├── train_models.py
    ├── generate_visualizations.py
    ├── data/
    │   └── Overall Data.csv
    ├── models/
    │   ├── *.pkl (6 models)
    │   ├── scaler.pkl
    │   ├── training_results.csv
    │   ├── test_results.csv
    │   └── test_predictions_*.csv
    └── visualizations/
        ├── *.png (8 files)
        └── all_results.xlsx
    """
    
    ax.text(8, 8.3, files_text, ha='left', va='top', fontsize=6.5, 
            color='#2c3e50', family='monospace')
    
    # Right side: Tech stack
    ax.text(8, 3.8, 'TECH STACK', ha='left', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.add_patch(FancyBboxPatch((7.5, 1.5), 6, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    
    tech_text = """
    • Python 3.13.7
    • scikit-learn (ML models)
    • XGBoost, LightGBM
    • pandas, numpy
    • matplotlib, seaborn
    • openpyxl (Excel)
    • joblib (Model saving)
    """
    
    ax.text(8, 3.3, tech_text, ha='left', va='top', fontsize=7, color='#2c3e50')
    
    plt.tight_layout()
    plt.savefig('/home/inanotherlife/Mining ANN/new/visualizations/FLOW_DIAGRAM.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Flow diagram saved!")
    plt.close()


def create_data_flow_diagram():
    """Create data transformation flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'DATA TRANSFORMATION PIPELINE', 
            ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.text(8, 9, 'From Raw CSV to Model Predictions',
            ha='center', fontsize=11, style='italic', color='#34495e')
    
    # Stage 1: Raw Data
    ax.add_patch(FancyBboxPatch((0.5, 7), 2.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#3498db', edgecolor='black', linewidth=2))
    ax.text(1.75, 7.8, 'RAW CSV', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(1.75, 7.5, 'Overall Data.csv', ha='center', fontsize=7, color='white')
    ax.text(1.75, 7.25, '361 samples', ha='center', fontsize=7, color='white')
    
    ax.arrow(3.1, 7.6, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Stage 2: Parsed Data
    ax.add_patch(FancyBboxPatch((4.2, 6.8), 2.5, 1.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#2ecc71', edgecolor='black', linewidth=2))
    ax.text(5.45, 8.1, 'PARSED DATA', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(5.45, 7.7, 'Features (X):', ha='center', fontsize=7, color='white')
    ax.text(5.45, 7.45, '• Cohesion', ha='center', fontsize=6.5, color='white')
    ax.text(5.45, 7.25, '• Friction Angle', ha='center', fontsize=6.5, color='white')
    ax.text(5.45, 7.05, '• Unit Weight', ha='center', fontsize=6.5, color='white')
    ax.text(5.45, 6.85, '• Ru (optional)', ha='center', fontsize=6.5, color='white')
    
    ax.arrow(6.8, 7.6, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Stage 3: Split Data
    ax.add_patch(FancyBboxPatch((7.9, 7), 2.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#9b59b6', edgecolor='black', linewidth=2))
    ax.text(9.15, 7.8, 'SPLIT DATA', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(9.15, 7.5, '80/20 Split', ha='center', fontsize=7, color='white')
    ax.text(9.15, 7.25, 'Stratified', ha='center', fontsize=7, color='white')
    
    # Branch to train and test
    ax.arrow(9.15, 6.9, -1.5, -0.7, head_width=0.12, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    ax.arrow(9.15, 6.9, 1.5, -0.7, head_width=0.12, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    
    # Training branch
    y_train = 5.5
    ax.add_patch(FancyBboxPatch((5.5, y_train), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#27ae60', edgecolor='black', linewidth=2))
    ax.text(6.75, y_train+0.65, 'TRAIN SET', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(6.75, y_train+0.35, '288 samples (80%)', ha='center', fontsize=7, color='white')
    ax.text(6.75, y_train+0.1, 'Shape: (288, 4)', ha='center', fontsize=6.5, color='white')
    
    ax.arrow(6.75, y_train-0.1, 0, -0.6, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Testing branch
    y_test = 5.5
    ax.add_patch(FancyBboxPatch((9, y_test), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#e67e22', edgecolor='black', linewidth=2))
    ax.text(10.25, y_test+0.65, 'TEST SET', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(10.25, y_test+0.35, '73 samples (20%)', ha='center', fontsize=7, color='white')
    ax.text(10.25, y_test+0.1, 'Shape: (73, 4)', ha='center', fontsize=6.5, color='white')
    
    # Stage 4: Scaling
    y_scale = 4.2
    ax.add_patch(FancyBboxPatch((5.5, y_scale), 2.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#2980b9', edgecolor='black', linewidth=2))
    ax.text(6.75, y_scale+0.4, 'STANDARDSCALER', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(6.75, y_scale+0.1, 'Fit & Transform', ha='center', fontsize=6.5, color='white')
    
    # Apply scaler to test
    ax.arrow(8.1, y_scale+0.4, 0.8, 0, head_width=0.1, head_length=0.08, 
             fc='#e67e22', ec='#e67e22', linewidth=1.5, linestyle='--')
    ax.text(8.5, y_scale+0.65, 'Transform Only', ha='center', fontsize=6, style='italic', color='#e67e22')
    
    ax.add_patch(FancyBboxPatch((9, y_scale), 2.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#2980b9', edgecolor='black', linewidth=2))
    ax.text(10.25, y_scale+0.4, 'STANDARDSCALER', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(10.25, y_scale+0.1, 'Transform Only', ha='center', fontsize=6.5, color='white')
    
    ax.arrow(6.75, y_scale-0.1, 0, -0.6, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Stage 5: Model Training
    y_model = 2.8
    ax.add_patch(FancyBboxPatch((5, y_model), 3.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#e74c3c', edgecolor='black', linewidth=2.5))
    ax.text(6.75, y_model+0.65, '6 MODELS TRAINING', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(6.75, y_model+0.35, 'Parallel Training', ha='center', fontsize=7, color='white')
    ax.text(6.75, y_model+0.1, 'on Scaled Train Data', ha='center', fontsize=6.5, color='white')
    
    ax.arrow(6.75, y_model-0.1, 0, -0.6, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Stage 6: Predictions
    y_pred = 1.4
    ax.add_patch(FancyBboxPatch((5, y_pred), 3.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(6.75, y_pred+0.4, 'TRAIN PREDICTIONS', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(6.75, y_pred+0.1, 'Shape: (288, 6 models)', ha='center', fontsize=6.5, color='white')
    
    # Test predictions (best 2 only)
    ax.arrow(10.25, y_scale-0.1, 0, -1.3, head_width=0.15, head_length=0.1, 
             fc='#e67e22', ec='#e67e22', linewidth=2, linestyle='--')
    
    ax.add_patch(FancyBboxPatch((9, y_pred), 2.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFD700', edgecolor='black', linewidth=2))
    ax.text(10.25, y_pred+0.4, 'TEST PREDICTIONS', ha='center', fontsize=8, fontweight='bold', color='#2c3e50')
    ax.text(10.25, y_pred+0.1, 'Shape: (73, 2 models)', ha='center', fontsize=6.5, color='#2c3e50')
    
    ax.arrow(6.75, y_pred-0.1, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.arrow(10.25, y_pred-0.1, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Stage 7: Outputs
    y_out = 0.3
    ax.add_patch(FancyBboxPatch((3, y_out), 10, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#34495e', edgecolor='black', linewidth=2.5))
    ax.text(8, y_out+0.3, 'OUTPUTS: Models (.pkl) | CSV Results | Excel File | PNG Visualizations | JSON Metadata', 
            ha='center', fontsize=8, fontweight='bold', color='white')
    
    # Left side: Data dimensions
    ax.text(0.5, 5, 'DATA DIMENSIONS', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
    
    dims_text = """
    Raw CSV:
    • Rows: 361
    • Columns: Variable
    
    Parsed Features (X):
    • Shape: (361, 4)
    • dtypes: float64
    
    Train Set:
    • X_train: (288, 4)
    • y_train: (288,)
    
    Test Set:
    • X_test: (73, 4)
    • y_test: (73,)
    
    Scaled:
    • Mean=0, Std=1
    • Same shape
    """
    
    ax.text(0.5, 4.7, dims_text, ha='left', va='top', fontsize=7, 
            color='#2c3e50', family='monospace')
    
    # Right side: Transformation details
    ax.text(12.5, 5, 'TRANSFORMATIONS', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
    
    trans_text = """
    StandardScaler Formula:
    z = (x - μ) / σ
    
    Where:
    • x = feature value
    • μ = mean (from training)
    • σ = std dev (from training)
    
    Applied to:
    1. Cohesion (kPa)
    2. Friction Angle (°)
    3. Unit Weight (kN/m³)
    4. Ru (0-1 range)
    
    Benefits:
    • Equal feature importance
    • Faster convergence
    • Better model performance
    """
    
    ax.text(12.5, 4.7, trans_text, ha='left', va='top', fontsize=7, 
            color='#2c3e50', family='monospace')
    
    plt.tight_layout()
    plt.savefig('/home/inanotherlife/Mining ANN/new/visualizations/DATA_FLOW_DIAGRAM.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Data flow diagram saved!")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING ARCHITECTURE & FLOW DIAGRAMS")
    print("="*60 + "\n")
    
    print("Creating diagrams...")
    create_architecture_diagram()
    create_flow_diagram()
    create_data_flow_diagram()
    
    print("\n" + "="*60)
    print("✅ ALL DIAGRAMS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nSaved files:")
    print("1. ARCHITECTURE_DIAGRAM.png - System architecture overview")
    print("2. FLOW_DIAGRAM.png - Detailed process flow (18 steps)")
    print("3. DATA_FLOW_DIAGRAM.png - Data transformation pipeline")
    print("\nLocation: /home/inanotherlife/Mining ANN/new/visualizations/")
