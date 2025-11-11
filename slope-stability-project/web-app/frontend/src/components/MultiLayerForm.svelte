<script>
  import { createEventDispatcher } from 'svelte';
  import axios from 'axios';
  
  export let selectedModel = 'gradient_boosting';
  
  const dispatch = createEventDispatcher();
  
  // Mineral layer data structure
  let layers = [
    {
      id: 1,
      name: 'Laterite',
      cohesion: 23.33,
      frictionAngle: 23.07,
      unitWeight: 22.35,
      ru: 0.25
    },
    {
      id: 2,
      name: 'Phyllitic Clay',
      cohesion: 15.73,
      frictionAngle: 19.70,
      unitWeight: 18.76,
      ru: 0.35
    }
  ];
  
  let nextId = 3;
  
  // State
  let isLoading = false;
  let error = null;
  let showSingleLayer = false;
  
  // API URL
  const API_URL = 'http://localhost:5000';
  
  function addLayer() {
    layers = [...layers, {
      id: nextId++,
      name: `Layer ${layers.length + 1}`,
      cohesion: 20,
      frictionAngle: 25,
      unitWeight: 20,
      ru: 0.3
    }];
  }
  
  function removeLayer(id) {
    if (layers.length > 1) {
      layers = layers.filter(layer => layer.id !== id);
    }
  }
  
  async function handleSubmit() {
    error = null;
    isLoading = true;
    dispatch('loading', true);
    
    try {
      const requestData = {
        layers: layers.map(layer => ({
          name: layer.name,
          cohesion: parseFloat(layer.cohesion),
          friction_angle: parseFloat(layer.frictionAngle),
          unit_weight: parseFloat(layer.unitWeight),
          ru: parseFloat(layer.ru)
        })),
        model: selectedModel
      };
      
      const response = await axios.post(`${API_URL}/predict`, requestData);
      dispatch('prediction', response.data);
    } catch (err) {
      error = err.response?.data?.message || err.message || 'Prediction failed';
      dispatch('prediction', null);
    } finally {
      isLoading = false;
      dispatch('loading', false);
    }
  }
  
  function loadSampleData() {
    layers = [
      {
        id: 1,
        name: 'Laterite',
        cohesion: 23.33,
        frictionAngle: 23.07,
        unitWeight: 22.35,
        ru: 0.25
      },
      {
        id: 2,
        name: 'Phyllitic Clay',
        cohesion: 15.73,
        frictionAngle: 19.70,
        unitWeight: 18.76,
        ru: 0.35
      }
    ];
    nextId = 3;
  }
</script>

<div class="multi-layer-form">
  <div class="header-section">
    <h2>🪨 Mineral Layer Properties</h2>
    <p class="form-description">Enter properties for each mineral/soil layer</p>
    
    <div class="quick-actions">
      <button type="button" class="btn-sample" on:click={loadSampleData}>
        📋 Load Sample Data
      </button>
      <button type="button" class="btn-add" on:click={addLayer}>
        ➕ Add Layer
      </button>
    </div>
  </div>
  
  <div class="info-banner">
    <span class="info-icon">ℹ️</span>
    <div class="info-text">
      <strong>Quick Start:</strong> Enter material properties for each layer. The best model is pre-selected for highest accuracy.
    </div>
  </div>
  
  <form on:submit|preventDefault={handleSubmit}>
    <div class="layers-container">
      {#each layers as layer, index (layer.id)}
        <div class="layer-card" class:first={index === 0}>
          <div class="layer-header">
            <input 
              type="text" 
              bind:value={layer.name} 
              class="layer-name-input"
              placeholder="Layer name"
            />
            {#if layers.length > 1}
              <button 
                type="button" 
                class="btn-remove" 
                on:click={() => removeLayer(layer.id)}
                title="Remove layer"
              >
                ❌
              </button>
            {/if}
          </div>
          
          <div class="layer-properties">
            <!-- Cohesion -->
            <div class="form-group compact">
              <label>
                <span class="label-text">Cohesion (c)</span>
                <span class="label-unit">kPa</span>
              </label>
              <div class="input-row">
                <input 
                  type="number" 
                  bind:value={layer.cohesion}
                  min="0" 
                  max="100" 
                  step="0.01"
                  class="number-input-compact"
                  placeholder="0-100"
                />
                <input 
                  type="range" 
                  bind:value={layer.cohesion}
                  min="0" 
                  max="100" 
                  step="0.1"
                  class="range-input-inline"
                />
              </div>
            </div>
            
            <!-- Friction Angle -->
            <div class="form-group compact">
              <label>
                <span class="label-text">Friction Angle (φ)</span>
                <span class="label-unit">degrees</span>
              </label>
              <div class="input-row">
                <input 
                  type="number" 
                  bind:value={layer.frictionAngle}
                  min="0" 
                  max="45" 
                  step="0.01"
                  class="number-input-compact"
                  placeholder="0-45"
                />
                <input 
                  type="range" 
                  bind:value={layer.frictionAngle}
                  min="0" 
                  max="45" 
                  step="0.1"
                  class="range-input-inline"
                />
              </div>
            </div>
            
            <!-- Unit Weight -->
            <div class="form-group compact">
              <label>
                <span class="label-text">Unit Weight (γ)</span>
                <span class="label-unit">kN/m³</span>
              </label>
              <div class="input-row">
                <input 
                  type="number" 
                  bind:value={layer.unitWeight}
                  min="15" 
                  max="25" 
                  step="0.01"
                  class="number-input-compact"
                  placeholder="15-25"
                />
                <input 
                  type="range" 
                  bind:value={layer.unitWeight}
                  min="15" 
                  max="25" 
                  step="0.1"
                  class="range-input-inline"
                />
              </div>
            </div>
            
            <!-- Ru -->
            <div class="form-group compact highlight-ru">
              <label>
                <span class="label-text">Pore Pressure Ratio (Ru)</span>
                <span class="label-badge">CRITICAL</span>
              </label>
              <div class="input-row">
                <input 
                  type="number" 
                  bind:value={layer.ru}
                  min="0" 
                  max="1" 
                  step="0.01"
                  class="number-input-compact ru-input"
                  placeholder="0-1"
                />
                <input 
                  type="range" 
                  bind:value={layer.ru}
                  min="0" 
                  max="1" 
                  step="0.01"
                  class="range-input-inline ru-slider"
                />
              </div>
            </div>
          </div>
        </div>
      {/each}
    </div>
    
    {#if error}
      <div class="error-message">
        <span class="error-icon">⚠️</span>
        {error}
      </div>
    {/if}
    
    <button 
      type="submit" 
      class="submit-button" 
      disabled={isLoading}
      class:loading={isLoading}
    >
      {#if isLoading}
        <span class="spinner"></span>
        Calculating...
      {:else}
        🎯 Predict Factor of Safety
      {/if}
    </button>
  </form>
</div>

<style>
  .multi-layer-form {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }
  
  .header-section {
    margin-bottom: 20px;
  }
  
  h2 {
    color: #2c3e50;
    margin: 0 0 8px 0;
    font-size: 1.5em;
    font-weight: 600;
  }
  
  .form-description {
    color: #7f8c8d;
    margin: 0 0 16px 0;
    font-size: 0.95em;
  }
  
  .quick-actions {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }
  
  .btn-sample, .btn-add {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.9em;
  }
  
  .btn-sample {
    background: #e8f5e9;
    color: #2e7d32;
  }
  
  .btn-sample:hover {
    background: #c8e6c9;
  }
  
  .btn-add {
    background: #3498db;
    color: white;
  }
  
  .btn-add:hover {
    background: #2980b9;
  }
  
  .info-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .info-icon {
    font-size: 1.5em;
  }
  
  .info-text {
    flex: 1;
    font-size: 0.9em;
    line-height: 1.4;
  }
  
  .layers-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 20px;
  }
  
  .layer-card {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 16px;
    transition: all 0.3s;
  }
  
  .layer-card:hover {
    border-color: #3498db;
    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.1);
  }
  
  .layer-card.first {
    border-color: #3498db;
  }
  
  .layer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }
  
  .layer-name-input {
    flex: 1;
    font-size: 1.1em;
    font-weight: 600;
    color: #2c3e50;
    border: none;
    background: transparent;
    padding: 4px 8px;
    border-bottom: 2px solid transparent;
    transition: border-color 0.2s;
  }
  
  .layer-name-input:hover {
    border-bottom-color: #3498db;
  }
  
  .layer-name-input:focus {
    outline: none;
    border-bottom-color: #3498db;
  }
  
  .btn-remove {
    background: none;
    border: none;
    font-size: 1.2em;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.2s;
    padding: 4px 8px;
  }
  
  .btn-remove:hover {
    opacity: 1;
  }
  
  .layer-properties {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  
  .form-group.compact {
    margin: 0;
  }
  
  .form-group.compact label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.9em;
  }
  
  .label-text {
    color: #34495e;
    font-weight: 500;
  }
  
  .label-unit {
    color: #95a5a6;
    font-size: 0.85em;
  }
  
  .label-badge {
    background: #e74c3c;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 600;
  }
  
  .input-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  
  .number-input-compact {
    width: 80px;
    padding: 8px 12px;
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    font-size: 0.95em;
    font-weight: 500;
    text-align: center;
  }
  
  .number-input-compact:focus {
    outline: none;
    border-color: #3498db;
  }
  
  .range-input-inline {
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    background: linear-gradient(to right, #e0e0e0 0%, #3498db 50%, #e0e0e0 100%);
    border-radius: 3px;
    outline: none;
  }
  
  .range-input-inline::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    background: #3498db;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  
  .range-input-inline::-moz-range-thumb {
    width: 18px;
    height: 18px;
    background: #3498db;
    border-radius: 50%;
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  
  .highlight-ru {
    grid-column: 1 / -1;
  }
  
  .highlight-ru label {
    background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 12px;
  }
  
  .ru-input {
    border-color: #f39c12;
  }
  
  .ru-input:focus {
    border-color: #e67e22;
    box-shadow: 0 0 0 3px rgba(243, 156, 18, 0.1);
  }
  
  .ru-slider {
    background: linear-gradient(to right, #e8f5e9 0%, #e74c3c 100%);
  }
  
  .ru-slider::-webkit-slider-thumb {
    background: #f39c12;
  }
  
  .ru-slider::-moz-range-thumb {
    background: #f39c12;
  }
  
  .error-message {
    background: #ffe5e5;
    color: #c62828;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-left: 4px solid #c62828;
  }
  
  .error-icon {
    font-size: 1.2em;
  }
  
  .submit-button {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1.1em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }
  
  .submit-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
  }
  
  .submit-button:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .submit-button.loading {
    background: #95a5a6;
  }
  
  .spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    margin-right: 8px;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  @media (max-width: 768px) {
    .layer-properties {
      grid-template-columns: 1fr;
    }
    
    .quick-actions {
      flex-direction: column;
    }
    
    .btn-sample, .btn-add {
      width: 100%;
    }
  }
</style>
