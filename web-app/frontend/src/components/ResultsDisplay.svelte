<script>
  export let result;
  
  const statusConfig = {
    'CRITICAL': { color: 'red', text: 'Slope failure likely' },
    'WARNING': { color: 'orange', text: 'Marginal stability' },
    'CAUTION': { color: 'yellow', text: 'Requires monitoring' },
    'SAFE': { color: 'green', text: 'Stable slope' }
  };
  
  $: safetyStatus = result?.safety?.status || '';
  $: safetyClass = result?.safety?.color || 'gray';
  $: safetyMessage = result?.safety?.message || '';
  $: fos = result?.prediction?.fos || 0;
  $: confidenceLower = result?.prediction?.confidence_interval?.lower || 0;
  $: confidenceUpper = result?.prediction?.confidence_interval?.upper || 0;
  $: config = statusConfig[safetyStatus] || { color: 'gray', text: 'Unknown status' };
</script>

<div class="results-display">
  <h2>📊 Prediction Results</h2>
  
  {#if result?.success}
    <!-- Main FoS Display -->
    <div class="fos-card {safetyClass}">
      <div class="fos-label">Factor of Safety (FoS)</div>
      <div class="fos-value">{fos.toFixed(4)}</div>
      <div class="confidence-interval">
        95% CI: [{confidenceLower.toFixed(4)}, {confidenceUpper.toFixed(4)}]
      </div>
    </div>
    
    <!-- Safety Status -->
    <div class="result-card status-card {safetyClass}">
      <div class="status-content">
        <div class="status-label">Safety Status</div>
        <div class="status-value">{safetyStatus}</div>
        <div class="status-description">{safetyMessage}</div>
      </div>
    </div>
    
    <!-- Input Parameters -->
    <div class="parameters-card">
      <h3>Input Parameters</h3>
      <div class="parameters-grid">
        <div class="param-item">
          <span class="param-label">Cohesion</span>
          <span class="param-value">{result.inputs.cohesion} kPa</span>
        </div>
        <div class="param-item">
          <span class="param-label">Friction Angle</span>
          <span class="param-value">{result.inputs.friction_angle}°</span>
        </div>
        <div class="param-item">
          <span class="param-label">Unit Weight</span>
          <span class="param-value">{result.inputs.unit_weight} kN/m³</span>
        </div>
        <div class="param-item highlight">
          <span class="param-label">Ru (Pore Pressure)</span>
          <span class="param-value">{result.inputs.ru}</span>
        </div>
      </div>
    </div>
    
    <!-- Model Information -->
    <div class="model-card">
      <h3>Model: {result.model.name}</h3>
      <div class="model-metrics">
        <div class="metric">
          <span class="metric-label">R² Score</span>
          <span class="metric-value">{(result.model.r2_score * 100).toFixed(2)}%</span>
        </div>
        <div class="metric">
          <span class="metric-label">RMSE</span>
          <span class="metric-value">{result.model.rmse.toFixed(4)}</span>
        </div>
        <div class="metric">
          <span class="metric-label">MAE</span>
          <span class="metric-value">{result.model.mae.toFixed(4)}</span>
        </div>
      </div>
    </div>
    
    <!-- FoS Interpretation Guide -->
    <div class="guide-card">
      <h3>📖 FoS Interpretation Guide</h3>
      <div class="guide-items">
        <div class="guide-item">
          <span class="guide-marker red">●</span>
          <span class="guide-text"><strong>FoS &lt; 1.0:</strong> Critical - Slope is unstable</span>
        </div>
        <div class="guide-item">
          <span class="guide-marker orange">●</span>
          <span class="guide-text"><strong>1.0 ≤ FoS &lt; 1.3:</strong> Warning - Marginal stability</span>
        </div>
        <div class="guide-item">
          <span class="guide-marker yellow">●</span>
          <span class="guide-text"><strong>1.3 ≤ FoS &lt; 1.5:</strong> Caution - Monitor conditions</span>
        </div>
        <div class="guide-item">
          <span class="guide-marker green">●</span>
          <span class="guide-text"><strong>FoS ≥ 1.5:</strong> Safe - Slope is stable</span>
        </div>
      </div>
    </div>
  {:else}
    <div class="no-results">
      <p>No prediction results yet. Fill in the form and click "Predict".</p>
    </div>
  {/if}
</div>

<style lang="css">
  .results-display {
    width: 100%;
  }
  
  .results-display h2 {
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    color: var(--text-primary);
  }
  
  .results-display h3 {
    font-size: 1.1rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
  }
  
  .fos-card {
    text-align: center;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 3px solid;
  }
  
  .fos-card.green {
    background: #d1fae5;
    border-color: var(--secondary-color);
  }
  
  .fos-card.yellow {
    background: #fef3c7;
    border-color: var(--warning-color);
  }
  
  .fos-card.orange {
    background: #fed7aa;
    border-color: #ea580c;
  }
  
  .fos-card.red {
    background: #fee2e2;
    border-color: var(--danger-color);
  }
  
  .fos-label {
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
  }
  
  .fos-value {
    font-size: 3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
  }
  
  .confidence-interval {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-family: 'Courier New', monospace;
  }
  
  .safety-status {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 2px solid;
  }
  
  .safety-status.green {
    background: #d1fae5;
    border-color: var(--secondary-color);
  }
  
  .safety-status.yellow {
    background: #fef3c7;
    border-color: var(--warning-color);
  }
  
  .safety-status.orange {
    background: #fed7aa;
    border-color: #ea580c;
  }
  
  .safety-status.red {
    background: #fee2e2;
    border-color: var(--danger-color);
  }
  
  .status-content {
    flex: 1;
  }
  
  .status-label {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }
  
  .status-message {
    font-size: 0.9rem;
    color: var(--text-secondary);
  }
  
  .parameters-card,
  .model-card,
  .guide-card {
    background: var(--bg-color);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
  }
  
  .parameters-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }
  
  .param-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: white;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }
  
  .param-item.highlight {
    background: #eff6ff;
    border-color: #3b82f6;
    font-weight: 600;
  }
  
  .param-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  
  .param-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .model-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }
  
  .metric {
    text-align: center;
    padding: 1rem;
    background: white;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }
  
  .metric-label {
    display: block;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
  }
  
  .metric-value {
    display: block;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary-color);
  }
  
  .guide-items {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .guide-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: white;
    border-radius: 6px;
  }
  
  .guide-marker {
    font-size: 1.25rem;
  }
  
  .guide-marker.red { color: var(--danger-color); }
  .guide-marker.orange { color: #ea580c; }
  .guide-marker.yellow { color: var(--warning-color); }
  .guide-marker.green { color: var(--secondary-color); }
  
  .guide-text {
    font-size: 0.9rem;
    color: var(--text-primary);
  }
  
  .no-results {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-secondary);
  }
  
  @media (max-width: 768px) {
    .parameters-grid,
    .model-metrics {
      grid-template-columns: 1fr;
    }
    
    .fos-value {
      font-size: 2.5rem;
    }
  }
</style>
