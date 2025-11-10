<script>
  import { createEventDispatcher } from 'svelte';
  import axios from 'axios';
  
  export let selectedModel = 'gradient_boosting';
  
  const dispatch = createEventDispatcher();
  
  // Form data
  let cohesion = 25;
  let frictionAngle = 30;
  let unitWeight = 20;
  let ru = 0.0;
  
  // State
  let isLoading = false;
  let error = null;
  
  // API URL
  const API_URL = 'http://localhost:5000';
  
  async function handleSubmit() {
    error = null;
    isLoading = true;
    dispatch('loading', true);
    
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        cohesion: parseFloat(cohesion),
        friction_angle: parseFloat(frictionAngle),
        unit_weight: parseFloat(unitWeight),
        ru: parseFloat(ru),
        model: selectedModel
      });
      
      dispatch('prediction', response.data);
    } catch (err) {
      error = err.response?.data?.message || err.message || 'Prediction failed';
      dispatch('prediction', null);
    } finally {
      isLoading = false;
      dispatch('loading', false);
    }
  }
  
  function handleModelChange() {
    dispatch('modelChange', selectedModel);
  }
</script>

<div class="prediction-form">
  <h2>🔢 Input Parameters</h2>
  <p class="form-description">Enter soil parameters for Factor of Safety prediction</p>
  
  <form on:submit|preventDefault={handleSubmit}>
    <!-- Cohesion -->
    <div class="form-group">
      <label for="cohesion">
        <span class="label-text">Cohesion (c)</span>
        <span class="label-unit">kPa</span>
      </label>
      <div class="input-with-value">
        <input 
          type="range" 
          id="cohesion" 
          bind:value={cohesion}
          min="0" 
          max="100" 
          step="0.1"
          class="range-input"
        />
        <input 
          type="number" 
          bind:value={cohesion}
          min="0" 
          max="100" 
          step="0.1"
          class="number-input"
        />
      </div>
      <div class="range-labels">
        <span>0</span>
        <span>100</span>
      </div>
    </div>
    
    <!-- Friction Angle -->
    <div class="form-group">
      <label for="friction">
        <span class="label-text">Friction Angle (φ)</span>
        <span class="label-unit">degrees</span>
      </label>
      <div class="input-with-value">
        <input 
          type="range" 
          id="friction" 
          bind:value={frictionAngle}
          min="0" 
          max="45" 
          step="0.1"
          class="range-input"
        />
        <input 
          type="number" 
          bind:value={frictionAngle}
          min="0" 
          max="45" 
          step="0.1"
          class="number-input"
        />
      </div>
      <div class="range-labels">
        <span>0°</span>
        <span>45°</span>
      </div>
    </div>
    
    <!-- Unit Weight -->
    <div class="form-group">
      <label for="weight">
        <span class="label-text">Unit Weight (γ)</span>
        <span class="label-unit">kN/m³</span>
      </label>
      <div class="input-with-value">
        <input 
          type="range" 
          id="weight" 
          bind:value={unitWeight}
          min="15" 
          max="25" 
          step="0.1"
          class="range-input"
        />
        <input 
          type="number" 
          bind:value={unitWeight}
          min="15" 
          max="25" 
          step="0.1"
          class="number-input"
        />
      </div>
      <div class="range-labels">
        <span>15</span>
        <span>25</span>
      </div>
    </div>
    
    <!-- Ru (Pore Pressure Ratio) -->
    <div class="form-group highlight">
      <label for="ru">
        <span class="label-text">Ru (Pore Pressure Ratio)</span>
        <span class="label-unit">0-1</span>
      </label>
      <div class="input-with-value">
        <input 
          type="range" 
          id="ru" 
          bind:value={ru}
          min="0" 
          max="1" 
          step="0.01"
          class="range-input"
        />
        <input 
          type="number" 
          bind:value={ru}
          min="0" 
          max="1" 
          step="0.01"
          class="number-input"
        />
      </div>
      <div class="range-labels">
        <span>0.0 (Dry)</span>
        <span>1.0 (Saturated)</span>
      </div>
      <p class="field-note">💧 Ru = u/(γ×h) where u is pore water pressure</p>
    </div>
    
    <!-- Model Selection -->
    <div class="form-group">
      <label for="model">
        <span class="label-text">Prediction Model</span>
      </label>
      <select 
        id="model" 
        bind:value={selectedModel}
        on:change={handleModelChange}
        class="select-input"
      >
        <option value="gradient_boosting">Gradient Boosting (R²=0.9426)</option>
        <option value="xgboost">XGBoost (R²=0.9420)</option>
      </select>
    </div>
    
    {#if error}
      <div class="error-message">
        ❌ {error}
      </div>
    {/if}
    
    <button 
      type="submit" 
      class="submit-button"
      disabled={isLoading}
    >
      {#if isLoading}
        <span class="loader"></span>
        Calculating...
      {:else}
        🎯 Predict Factor of Safety
      {/if}
    </button>
  </form>
</div>

<style lang="css">
  .prediction-form {
    width: 100%;
  }
  
  .prediction-form h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
  }
  
  .form-description {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-bottom: 2rem;
  }
  
  .form-group {
    margin-bottom: 2rem;
  }
  
  .form-group.highlight {
    background: #eff6ff;
    padding: 1.25rem;
    border-radius: 8px;
    border: 2px solid #3b82f6;
  }
  
  label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .label-text {
    font-size: 0.95rem;
  }
  
  .label-unit {
    font-size: 0.85rem;
    font-weight: 400;
    color: var(--text-secondary);
    background: var(--bg-color);
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
  }
  
  .input-with-value {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1rem;
    align-items: center;
  }
  
  .range-input {
    width: 100%;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--border-color);
    outline: none;
    border-radius: 3px;
  }
  
  .range-input::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    background: var(--primary-color);
    cursor: pointer;
    border-radius: 50%;
    box-shadow: var(--shadow);
  }
  
  .range-input::-moz-range-thumb {
    width: 20px;
    height: 20px;
    background: var(--primary-color);
    cursor: pointer;
    border-radius: 50%;
    border: none;
    box-shadow: var(--shadow);
  }
  
  .number-input {
    width: 90px;
    padding: 0.5rem;
    border: 2px solid var(--border-color);
    border-radius: 6px;
    font-size: 1rem;
    text-align: center;
    font-weight: 600;
  }
  
  .number-input:focus {
    outline: none;
    border-color: var(--primary-color);
  }
  
  .range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }
  
  .field-note {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #1e40af;
    font-style: italic;
  }
  
  .select-input {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.95rem;
    background: white;
    cursor: pointer;
  }
  
  .select-input:focus {
    outline: none;
    border-color: var(--primary-color);
  }
  
  .submit-button {
    width: 100%;
    padding: 1rem;
    background: linear-gradient(135deg, var(--primary-color) 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }
  
  .submit-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
  }
  
  .submit-button:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .loader {
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top: 2px solid white;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  .error-message {
    padding: 1rem;
    background: #fee;
    border: 1px solid var(--danger-color);
    border-radius: 8px;
    color: var(--danger-color);
    margin-bottom: 1rem;
    font-size: 0.9rem;
  }
</style>
