#!/usr/bin/env python3
"""Create an interactive HTML dashboard for ML model results."""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_performance_data():
    """Load model performance metrics."""
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / "models" / "model_performance.json", 'r') as f:
        return json.load(f)

def generate_dashboard_html(data):
    """Generate interactive HTML dashboard."""
    
    # Sort models by R² score
    sorted_data = sorted(data, key=lambda x: x['r2'], reverse=True)
    
    # Generate model cards HTML
    model_cards = ""
    for i, model_data in enumerate(sorted_data):
        rank = i + 1
        model_name = model_data['model'].replace('_', ' ').title()
        r2 = model_data['r2']
        rmse = model_data['rmse']
        mae = model_data['mae']
        cv_mean = model_data['cv_r2_mean']
        cv_std = model_data['cv_r2_std']
        
        # Determine rank badge
        if rank == 1:
            rank_badge = '🥇 RANK 1'
            rank_color = '#FFD700'
        elif rank == 2:
            rank_badge = '🥈 RANK 2'
            rank_color = '#C0C0C0'
        elif rank == 3:
            rank_badge = '🥉 RANK 3'
            rank_color = '#CD7F32'
        else:
            rank_badge = f'RANK {rank}'
            rank_color = '#95a5a6'
        
        # Determine performance color
        if r2 >= 0.9:
            perf_color = '#2ecc71'
            perf_text = 'EXCELLENT'
        elif r2 >= 0.7:
            perf_color = '#f39c12'
            perf_text = 'GOOD'
        elif r2 >= 0.5:
            perf_color = '#3498db'
            perf_text = 'MODERATE'
        else:
            perf_color = '#e74c3c'
            perf_text = 'POOR'
        
        model_cards += f'''
        <div class="model-card" style="border-left: 5px solid {rank_color};">
            <div class="rank-badge" style="background: {rank_color};">{rank_badge}</div>
            <h2>{model_name}</h2>
            <div class="performance-badge" style="background: {perf_color};">{perf_text}</div>
            
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value" style="color: {perf_color};">{r2:.4f}</div>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {max(0, min(100, (r2 + 1) * 50))}%; background: {perf_color};"></div>
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{rmse:.4f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">{mae:.4f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">CV R² Mean</div>
                    <div class="metric-value">{cv_mean:.4f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">CV Std Dev</div>
                    <div class="metric-value">{cv_std:.4f}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{r2 * 100:.2f}%</div>
                </div>
            </div>
        </div>
        '''
    
    # Generate comparison table
    table_rows = ""
    for i, model_data in enumerate(sorted_data):
        rank = i + 1
        model_name = model_data['model'].replace('_', ' ').title()
        r2 = model_data['r2']
        rmse = model_data['rmse']
        mae = model_data['mae']
        cv_mean = model_data['cv_r2_mean']
        cv_std = model_data['cv_r2_std']
        
        row_color = '#f8f9fa' if rank % 2 == 0 else 'white'
        
        table_rows += f'''
        <tr style="background: {row_color};">
            <td><strong>{rank}</strong></td>
            <td><strong>{model_name}</strong></td>
            <td style="color: #2ecc71; font-weight: bold;">{r2:.4f}</td>
            <td>{rmse:.4f}</td>
            <td>{mae:.4f}</td>
            <td>{cv_mean:.4f}</td>
            <td>{cv_std:.4f}</td>
        </tr>
        '''
    
    html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Performance Dashboard - Factor of Safety Prediction</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .header .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
        
        .stats-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-card .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2ecc71;
            margin-bottom: 10px;
        }}
        
        .stat-card .stat-label {{
            color: #7f8c8d;
            font-size: 1em;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }}
        
        .model-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .model-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .rank-badge {{
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .model-card h2 {{
            color: #2c3e50;
            margin-bottom: 15px;
            border: none;
            padding: 0;
        }}
        
        .performance-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
            margin-bottom: 20px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .metric-bar {{
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }}
        
        .metric-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: #3498db;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background: #e8f4f8 !important;
        }}
        
        .recommendation {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
        }}
        
        .recommendation h3 {{
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .recommendation ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .recommendation li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        
        .recommendation li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .gallery-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .gallery-item img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .gallery-item h3 {{
            color: #2c3e50;
            font-size: 1.1em;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .gallery {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 ML Model Performance Dashboard</h1>
            <div class="subtitle">Factor of Safety Prediction - Mining Stability Analysis</div>
            <div class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        </div>
        
        <div class="stats-overview">
            <div class="stat-card">
                <div class="stat-value">5</div>
                <div class="stat-label">ML Algorithms</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sorted_data[0]['r2']:.4f}</div>
                <div class="stat-label">Best R² Score (SVM)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">150</div>
                <div class="stat-label">Training Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">33</div>
                <div class="stat-label">Input Features</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Model Performance Rankings</h2>
            {model_cards}
            
            <div class="recommendation">
                <h3>🏆 Deployment Recommendations</h3>
                <ul>
                    <li><strong>PRIMARY:</strong> Support Vector Machine (SVM) - Highest accuracy (94.98%), smallest model size (59 KB), excellent stability</li>
                    <li><strong>SECONDARY:</strong> Random Forest - Strong accuracy (93.41%), provides feature importance, robust</li>
                    <li><strong>ALTERNATIVE:</strong> LightGBM - Fast training, competitive accuracy (91.92%), smallest inference time</li>
                    <li><strong>NOT RECOMMENDED:</strong> ANN - Poor performance (R² = -0.73), requires more data</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Detailed Comparison Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>R² Score</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>CV R² Mean</th>
                        <th>CV Std Dev</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📸 Visualization Gallery</h2>
            <p style="color: #7f8c8d; margin-bottom: 20px;">
                Note: Run <code>python visualizations/generate_plots.py</code> to generate visualization images.
            </p>
            <div class="gallery">
                <div class="gallery-item">
                    <img src="plots/model_comparison.png" alt="Model Comparison" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Model Comparison</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/model_ranking.png" alt="Model Ranking" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Model Rankings</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/cv_results.png" alt="CV Results" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Cross-Validation Results</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/error_distribution.png" alt="Error Distribution" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Error Distribution</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/feature_importance.png" alt="Feature Importance" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Feature Importance</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/training_metrics.png" alt="Training Metrics" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Training Metrics</h3>
                </div>
                <div class="gallery-item">
                    <img src="plots/dataset_overview.png" alt="Dataset Overview" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'400\' height=\'300\'%3E%3Crect fill=\'%23ecf0f1\' width=\'400\' height=\'300\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' fill=\'%237f8c8d\' font-size=\'20\'%3EGenerate plots first%3C/text%3E%3C/svg%3E'">
                    <h3>Dataset Overview</h3>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>ℹ️ About This Dashboard</h2>
            <p style="color: #7f8c8d; line-height: 1.8;">
                This dashboard presents the performance analysis of 5 machine learning algorithms trained to predict the Factor of Safety (FoS) 
                for mining stability analysis. The models were evaluated using R² score, RMSE, MAE, and 5-fold cross-validation. 
                Support Vector Machine (SVM) emerged as the best performer with an R² score of 0.9498, demonstrating 94.98% accuracy 
                in predicting slope stability.
            </p>
        </div>
    </div>
</body>
</html>
'''
    
    return html_template

def main():
    """Generate HTML dashboard."""
    print("\n" + "="*80)
    print("Creating Interactive ML Dashboard")
    print("="*80 + "\n")
    
    # Load performance data
    data = load_performance_data()
    
    # Generate HTML
    html_content = generate_dashboard_html(data)
    
    # Save to file
    output_dir = Path(__file__).parent
    dashboard_path = output_dir / "dashboard.html"
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Dashboard created successfully!")
    print(f"✓ Saved to: {dashboard_path}")
    print(f"\n📊 Open the dashboard in your browser:")
    print(f"   file://{dashboard_path.absolute()}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
