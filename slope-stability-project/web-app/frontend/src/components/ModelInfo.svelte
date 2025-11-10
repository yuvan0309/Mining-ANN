<script>
  export let model = 'gradient_boosting';
  
  const modelData = {
    gradient_boosting: {
      name: 'Gradient Boosting',
      testR2: 0.9426,
      testRMSE: 0.0834,
      testMAE: 0.0563,
      trainR2: 0.9954,
      gap: '5.28%',
      rank: '1st',
      description: 'Best performing model with highest test accuracy',
      features: [
        'Highest test R² (94.26%)',
        'Lowest MAE (0.0563)',
        'Best overall accuracy',
        'Recommended for critical predictions'
      ]
    },
    xgboost: {
      name: 'XGBoost',
      testR2: 0.9420,
      testRMSE: 0.0838,
      testMAE: 0.0597,
      trainR2: 0.9581,
      gap: '1.61%',
      rank: '2nd',
      description: 'Excellent generalization with minimal overfitting',
      features: [
        'Best generalization (1.61% gap)',
        'Fine-tuned regularization',
        'Very close to GB (0.06% diff)',
        'Excellent for varying conditions'
      ]
    }
  };
  
  $: currentModel = modelData[model] || modelData.gradient_boosting;
</script>

<div class="model-info">
  <div class="model-header">
    <div>
      <h2>{currentModel.name}</h2>
      <p class="model-rank">{currentModel.rank} Best Model</p>
    </div>
  </div>
  
  <p class="model-description">{currentModel.description}</p>
  
  <div class="metrics-section">
    <h3>Performance Metrics</h3>
    
    <div class="metric-row">
      <span class="metric-label">Test R² Score</span>
      <span class="metric-value highlight">{(currentModel.testR2 * 100).toFixed(2)}%</span>
    </div>
    
    <div class="metric-row">
      <span class="metric-label">Test RMSE</span>
      <span class="metric-value">{currentModel.testRMSE.toFixed(4)}</span>
    </div>
    
    <div class="metric-row">
      <span class="metric-label">Test MAE</span>
      <span class="metric-value">{currentModel.testMAE.toFixed(4)}</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="metric-row">
      <span class="metric-label">Training R²</span>
      <span class="metric-value">{(currentModel.trainR2 * 100).toFixed(2)}%</span>
    </div>
    
    <div class="metric-row">
      <span class="metric-label">Overfitting Gap</span>
      <span class="metric-value">{currentModel.gap}</span>
    </div>
  </div>
  
  <div class="features-section">
    <h3>Key Features</h3>
    <ul class="features-list">
      {#each currentModel.features as feature}
        <li>{feature}</li>
      {/each}
    </ul>
  </div>
  
  <div class="tech-stack">
    <h3>Tech Stack</h3>
    <div class="tech-badges">
      <span class="badge">Python 3.13</span>
      <span class="badge">scikit-learn</span>
      <span class="badge">XGBoost</span>
      <span class="badge">LightGBM</span>
      <span class="badge">Flask</span>
      <span class="badge">Svelte</span>
    </div>
  </div>
</div>

<style lang="css">
  .model-info {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color);
  }
  
  .model-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  
  .model-info h2 {
    font-size: 1.25rem;
    margin-bottom: 0.25rem;
    color: var(--text-primary);
  }
  
  .model-rank {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 600;
  }
  
  .model-description {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: var(--bg-color);
    border-radius: 8px;
    border-left: 4px solid var(--primary-color);
  }
  
  .metrics-section,
  .features-section,
  .tech-stack {
    margin-bottom: 1.5rem;
  }
  
  .model-info h3 {
    font-size: 1rem;
    margin-bottom: 0.75rem;
    color: var(--text-primary);
  }
  
  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border-color);
  }
  
  .metric-row:last-child {
    border-bottom: none;
  }
  
  .metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  
  .metric-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .metric-value.highlight {
    font-size: 1.1rem;
    color: var(--primary-color);
  }
  
  .divider {
    height: 1px;
    background: var(--border-color);
    margin: 0.5rem 0;
  }
  
  .features-list,
  .info-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  
  .features-list li,
  .info-list li {
    padding: 0.5rem 0 0.5rem 1.5rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    position: relative;
  }
  
  .features-list li::before {
    content: '✓';
    position: absolute;
    left: 0;
    color: var(--secondary-color);
    font-weight: 700;
  }
  
  .info-list li::before {
    content: '•';
    position: absolute;
    left: 0.25rem;
    color: var(--primary-color);
  }
  
  .tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .badge {
    display: inline-block;
    padding: 0.375rem 0.75rem;
    background: var(--primary-color);
    color: white;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }
</style>
