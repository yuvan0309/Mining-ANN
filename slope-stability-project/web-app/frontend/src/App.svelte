<script>
  import axios from 'axios';

  const API_URL = 'http://localhost:5000';
  
  const models = [
    { name: 'Gradient Boosting', value: 'gradient_boosting', r2: 0.9426, best: true },
    { name: 'XGBoost', value: 'xgboost', r2: 0.9420, best: false }
  ];

  let selectedModel = 'gradient_boosting';
  
  let layer1Cohesion = '23.33', layer1Friction = '23.07', layer1Weight = '22.35', layer1Ru = '0.0';
  let layer2Cohesion = '15.73', layer2Friction = '19.70', layer2Weight = '18.76', layer2Ru = '0.0';
  let layer3Cohesion = '35.00', layer3Friction = '28.50', layer3Weight = '24.00', layer3Ru = '0.0';
  let layer4Cohesion = '5.20', layer4Friction = '32.00', layer4Weight = '18.50', layer4Ru = '0.0';
  let layer5Cohesion = '18.60', layer5Friction = '22.40', layer5Weight = '19.80', layer5Ru = '0.0';
  let layer6Cohesion = '8.50', layer6Friction = '35.20', layer6Weight = '20.40', layer6Ru = '0.0';
  let layer7Cohesion = '28.40', layer7Friction = '31.80', layer7Weight = '21.60', layer7Ru = '0.0';
  let layer8Cohesion = '45.00', layer8Friction = '38.50', layer8Weight = '26.00', layer8Ru = '0.0';
  
  let prediction = null, loading = false, error = null;
  
  // Debug reactive statement
  $: if (prediction) {
    console.log('Prediction updated:', prediction);
  }

  async function handlePredict() {
    if (!layer1Cohesion || !layer1Friction || !layer1Weight) {
      error = 'Please fill in all required fields';
      return;
    }
    loading = true;
    error = null;
    try {
      const avgCohesion = (parseFloat(layer1Cohesion) + parseFloat(layer2Cohesion) + parseFloat(layer3Cohesion) + parseFloat(layer4Cohesion) + parseFloat(layer5Cohesion) + parseFloat(layer6Cohesion) + parseFloat(layer7Cohesion) + parseFloat(layer8Cohesion)) / 8;
      const avgFriction = (parseFloat(layer1Friction) + parseFloat(layer2Friction) + parseFloat(layer3Friction) + parseFloat(layer4Friction) + parseFloat(layer5Friction) + parseFloat(layer6Friction) + parseFloat(layer7Friction) + parseFloat(layer8Friction)) / 8;
      const avgWeight = (parseFloat(layer1Weight) + parseFloat(layer2Weight) + parseFloat(layer3Weight) + parseFloat(layer4Weight) + parseFloat(layer5Weight) + parseFloat(layer6Weight) + parseFloat(layer7Weight) + parseFloat(layer8Weight)) / 8;
      const avgRu = (parseFloat(layer1Ru) + parseFloat(layer2Ru) + parseFloat(layer3Ru) + parseFloat(layer4Ru) + parseFloat(layer5Ru) + parseFloat(layer6Ru) + parseFloat(layer7Ru) + parseFloat(layer8Ru)) / 8;
      
      console.log('Sending prediction request:', { avgCohesion, avgFriction, avgWeight, avgRu, selectedModel });
      
      const response = await axios.post(`${API_URL}/predict`, {
        cohesion: avgCohesion,
        friction_angle: avgFriction,
        unit_weight: avgWeight,
        ru: avgRu,
        model: selectedModel
      });
      
      console.log('Response received:', response.data);
      
      // The backend returns nested structure, extract the FoS value
      if (response.data.prediction && response.data.prediction.fos) {
        prediction = {
          fos: response.data.prediction.fos,
          model: response.data.model,
          inputs: response.data.inputs
        };
      } else {
        prediction = response.data;
      }
      
      console.log('Prediction set to:', prediction);
    } catch (err) {
      console.error('Prediction error:', err);
      error = err.response?.data?.error || 'Failed to get prediction';
    } finally {
      loading = false;
    }
  }

  function generateRandomValue(min, max, decimals = 2) {
    const value = Math.random() * (max - min) + min;
    return value.toFixed(decimals);
  }

  function loadSample() {
    // Laterite - typical ranges
    layer1Cohesion = generateRandomValue(20, 30);
    layer1Friction = generateRandomValue(20, 26);
    layer1Weight = generateRandomValue(20, 24);
    layer1Ru = generateRandomValue(0, 0.3);
    
    // Phyllitic Clay - lower cohesion, moderate friction
    layer2Cohesion = generateRandomValue(10, 20);
    layer2Friction = generateRandomValue(18, 22);
    layer2Weight = generateRandomValue(17, 20);
    layer2Ru = generateRandomValue(0.1, 0.4);
    
    // Lumpy Iron Ore - high cohesion and friction
    layer3Cohesion = generateRandomValue(30, 40);
    layer3Friction = generateRandomValue(26, 32);
    layer3Weight = generateRandomValue(22, 25);
    layer3Ru = generateRandomValue(0, 0.2);
    
    // Limonitic Clay - very low cohesion, high friction
    layer4Cohesion = generateRandomValue(3, 8);
    layer4Friction = generateRandomValue(28, 36);
    layer4Weight = generateRandomValue(17, 20);
    layer4Ru = generateRandomValue(0.2, 0.5);
    
    // Manganiferous Clay - moderate properties
    layer5Cohesion = generateRandomValue(15, 22);
    layer5Friction = generateRandomValue(20, 25);
    layer5Weight = generateRandomValue(18, 21);
    layer5Ru = generateRandomValue(0.1, 0.4);
    
    // Siliceous Clay - low cohesion, high friction
    layer6Cohesion = generateRandomValue(6, 12);
    layer6Friction = generateRandomValue(32, 38);
    layer6Weight = generateRandomValue(19, 22);
    layer6Ru = generateRandomValue(0.1, 0.4);
    
    // BHQ (Banded Hematite Quartzite) - high strength
    layer7Cohesion = generateRandomValue(25, 35);
    layer7Friction = generateRandomValue(28, 35);
    layer7Weight = generateRandomValue(20, 23);
    layer7Ru = generateRandomValue(0, 0.2);
    
    // Schist - very high strength
    layer8Cohesion = generateRandomValue(40, 50);
    layer8Friction = generateRandomValue(35, 42);
    layer8Weight = generateRandomValue(24, 27);
    layer8Ru = generateRandomValue(0, 0.15);
  }

  function resetForm() {
    // Clear all inputs
    layer1Cohesion = ''; layer1Friction = ''; layer1Weight = ''; layer1Ru = '';
    layer2Cohesion = ''; layer2Friction = ''; layer2Weight = ''; layer2Ru = '';
    layer3Cohesion = ''; layer3Friction = ''; layer3Weight = ''; layer3Ru = '';
    layer4Cohesion = ''; layer4Friction = ''; layer4Weight = ''; layer4Ru = '';
    layer5Cohesion = ''; layer5Friction = ''; layer5Weight = ''; layer5Ru = '';
    layer6Cohesion = ''; layer6Friction = ''; layer6Weight = ''; layer6Ru = '';
    layer7Cohesion = ''; layer7Friction = ''; layer7Weight = ''; layer7Ru = '';
    layer8Cohesion = ''; layer8Friction = ''; layer8Weight = ''; layer8Ru = '';
    
    // Clear prediction and error
    prediction = null;
    error = null;
  }

  function getFosStatus(fos) {
    if (fos > 1.5) return { text: 'Safe - Excellent stability', color: '#22c55e' };
    if (fos >= 1.2) return { text: 'WARNING - Monitor closely', color: '#f59e0b' };
    return { text: 'Unsafe - Critical condition', color: '#ef4444' };
  }

  $: selectedModelData = models.find(m => m.value === selectedModel);
  $: fosStatus = (prediction && prediction.fos) ? getFosStatus(prediction.fos) : null;
  
  // Debug - log when fosStatus changes
  $: if (fosStatus) {
    console.log('fosStatus updated:', fosStatus);
  }
</script>

<main>
  <header>
    <h1>Factor of Safety (FoS) Prediction System</h1>
    <p>Machine Learning-Based Mining Slope Stability Analysis</p>
  </header>

  <div class="container">
    <div class="left-panel">
      <section class="model-section">
        <h2>Select Prediction Model</h2>
        <div class="model-grid">
          {#each models as model}
            <button class="model-btn" class:selected={selectedModel === model.value} on:click={() => selectedModel = model.value}>
              <span class="model-name">{model.name}</span>
              <span class="model-r2">R² = {model.r2.toFixed(4)}</span>
              {#if model.best}<span class="best-badge">⭐ BEST</span>{/if}
            </button>
          {/each}
        </div>
      </section>

      <div class="info-tip">
        <strong>Quick Start:</strong> Click "🎲 Load Random Data" to generate sample values for all 8 layers, or enter values manually. Click "🔄 Reset" to clear all fields. Best performing model is pre-selected.
      </div>

      <section class="layer-section"><h2>Laterite</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer1Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer1Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer1Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer1Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Phyllitic Clay</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer2Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer2Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer2Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer2Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Lumpy Iron Ore</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer3Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer3Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer3Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer3Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Limonitic Clay</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer4Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer4Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer4Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer4Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Manganiferous Clay</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer5Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer5Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer5Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer5Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Siliceous Clay</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer6Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer6Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer6Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer6Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>BHQ</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer7Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer7Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer7Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer7Ru} /></label>
        </div>
      </section>

      <section class="layer-section"><h2>Schist</h2>
        <div class="input-grid">
          <label>Cohesion (kPa)<input type="number" step="0.01" bind:value={layer8Cohesion} /></label>
          <label>Friction (°)<input type="number" step="0.01" bind:value={layer8Friction} /></label>
          <label>Weight (kN/m³)<input type="number" step="0.01" bind:value={layer8Weight} /></label>
          <label>Ru<input type="number" step="0.01" min="0" max="1" bind:value={layer8Ru} /></label>
        </div>
      </section>

      <div class="btn-row">
        <button class="btn-secondary" on:click={loadSample}>🎲 Load Random Data</button>
        <button class="btn-reset" on:click={resetForm}>🔄 Reset</button>
        <button class="btn-primary" on:click={handlePredict} disabled={loading}>
          {loading ? '⏳ Calculating...' : 'Predict FoS'}
        </button>
      </div>

      {#if error}<div class="error">{error}</div>{/if}
    </div>

    <div class="right-panel">
      <section class="results-section">
        {#if prediction}
          <div class="prediction-result">
            <!-- Professional FoS Gauge -->
            <div class="gauge-container">
              <svg viewBox="0 0 240 140" class="gauge-svg">
                <defs>
                  <!-- Gradient definitions for smooth color transitions -->
                  <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#dc2626;stop-opacity:1" />
                    <stop offset="25%" style="stop-color:#ea580c;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#eab308;stop-opacity:1" />
                    <stop offset="75%" style="stop-color:#84cc16;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#22c55e;stop-opacity:1" />
                  </linearGradient>
                  
                  <!-- Shadow filter -->
                  <filter id="gaugeShadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
                    <feOffset dx="0" dy="2" result="offsetblur"/>
                    <feComponentTransfer>
                      <feFuncA type="linear" slope="0.3"/>
                    </feComponentTransfer>
                    <feMerge>
                      <feMergeNode/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                
                <!-- Outer ring background -->
                <path d="M 30 110 A 90 90 0 0 1 210 110" 
                      fill="none" 
                      stroke="#f1f5f9" 
                      stroke-width="24" 
                      stroke-linecap="round"/>
                
                <!-- Color zones with gradient -->
                <path d="M 30 110 A 90 90 0 0 1 210 110" 
                      fill="none" 
                      stroke="url(#gaugeGradient)" 
                      stroke-width="20" 
                      stroke-linecap="round"
                      filter="url(#gaugeShadow)"/>
                
                <!-- Tick marks at min and max only -->
                {#each [0, 2.5] as tick}
                  <line x1="{120 + 75 * Math.cos((Math.PI * (tick / 2.5 * 180 - 180)) / 180)}" 
                        y1="{110 + 75 * Math.sin((Math.PI * (tick / 2.5 * 180 - 180)) / 180)}"
                        x2="{120 + 85 * Math.cos((Math.PI * (tick / 2.5 * 180 - 180)) / 180)}" 
                        y2="{110 + 85 * Math.sin((Math.PI * (tick / 2.5 * 180 - 180)) / 180)}"
                        stroke="#64748b" 
                        stroke-width="2" 
                        stroke-linecap="round"/>
                {/each}
                
                <!-- Scale labels - Min and Max only -->
                <text x="30" y="125" font-size="13" font-weight="700" fill="#1e293b" text-anchor="middle">0</text>
                <text x="210" y="125" font-size="13" font-weight="700" fill="#1e293b" text-anchor="middle">2.5</text>
                
                <!-- Needle shadow -->
                <line x1="120" 
                      y1="110" 
                      x2="{120 + 68 * Math.cos((Math.PI * (Math.min(Math.max(prediction.fos, 0), 2.5) / 2.5 * 180 - 180)) / 180)}" 
                      y2="{110 + 68 * Math.sin((Math.PI * (Math.min(Math.max(prediction.fos, 0), 2.5) / 2.5 * 180 - 180)) / 180)}" 
                      stroke="#94a3b8" 
                      stroke-width="4" 
                      stroke-linecap="round"
                      opacity="0.3"/>
                
                <!-- Needle -->
                <line x1="120" 
                      y1="110" 
                      x2="{120 + 70 * Math.cos((Math.PI * (Math.min(Math.max(prediction.fos, 0), 2.5) / 2.5 * 180 - 180)) / 180)}" 
                      y2="{110 + 70 * Math.sin((Math.PI * (Math.min(Math.max(prediction.fos, 0), 2.5) / 2.5 * 180 - 180)) / 180)}" 
                      stroke="{fosStatus?.color || '#6b7280'}" 
                      stroke-width="3.5" 
                      stroke-linecap="round"
                      filter="url(#gaugeShadow)"/>
                
                <!-- Center circle background -->
                <circle cx="120" cy="110" r="10" fill="white" filter="url(#gaugeShadow)"/>
                
                <!-- Center circle -->
                <circle cx="120" cy="110" r="7" fill="{fosStatus?.color || '#6b7280'}"/>
                <circle cx="120" cy="110" r="3" fill="white" opacity="0.5"/>
              </svg>
              
              <div class="gauge-value">
                <div class="gauge-number" style="color: {fosStatus?.color || '#1f2937'}">{prediction.fos.toFixed(3)}</div>
                <div class="gauge-label">Factor of Safety</div>
              </div>
            </div>
            
            <div class="fos-status-box" style="border-left: 4px solid {fosStatus?.color || '#6b7280'}">
              <div class="status-text" style="color: {fosStatus?.color || '#6b7280'}">{fosStatus?.text || 'Status unknown'}</div>
            </div>
            <div class="model-used">
              <strong>Model:</strong> {selectedModelData.name}<br>
              <strong>Accuracy:</strong> R² = {selectedModelData.r2.toFixed(4)}
            </div>
            
            <div class="about-fos">
              <h3>About Factor of Safety (FoS)</h3>
              <p>The Factor of Safety indicates slope stability:</p>
              <ul>
                <li><strong>FoS &gt; 1.5:</strong> Safe for temporary and permanent benches</li>
                <li><strong>FoS = 1.3-1.5:</strong> Safe for temporary benches</li>
                <li><strong>FoS = 1.1-1.3:</strong> Monitor closely</li>
                <li><strong>FoS = 1.0-1.1:</strong> Unsafe</li>
                <li><strong>FoS &lt; 1.0:</strong> Dangerous - take immediate action</li>
              </ul>
            </div>
          </div>
        {:else}
          <div class="no-prediction">
            <div class="placeholder-icon">📊</div>
            <h3>No Prediction Yet</h3>
            <p>Enter material properties for all 8 layers and click "Predict FoS" to see results</p>
          </div>
        {/if}
      </section>
    </div>
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: linear-gradient(135deg, #5b7ef5 0%, #8b5cf6 50%, #a855f7 100%);
    min-height: 100vh;
  }
  main { max-width: 1600px; margin: 0 auto; padding: 20px; }
  header { text-align: center; color: white; margin-bottom: 30px; }
  h1 { font-size: 2.5rem; margin: 0 0 10px 0; font-weight: 700; }
  header p { font-size: 1.1rem; opacity: 0.95; margin: 0; }
  .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; height: calc(100vh - 180px); }
  .left-panel { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); overflow-y: auto; max-height: 100%; }
  .left-panel::-webkit-scrollbar { width: 8px; }
  .left-panel::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
  .left-panel::-webkit-scrollbar-thumb { background: #8b5cf6; border-radius: 10px; }
  .left-panel::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
  .right-panel { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); overflow-y: auto; }
  .model-section h2 { color: #1f2937; margin: 0 0 16px 0; font-size: 1.3rem; }
  .model-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 20px; }
  .model-btn { padding: 16px; border: 2px solid #e5e7eb; border-radius: 8px; background: white; cursor: pointer; transition: all 0.2s; text-align: left; position: relative; }
  .model-btn:hover { border-color: #8b5cf6; transform: translateY(-2px); }
  .model-btn.selected { border-color: #8b5cf6; background: linear-gradient(135deg, #f3f0ff 0%, #e9d5ff 100%); }
  .model-name { display: block; font-weight: 600; color: #1f2937; margin-bottom: 4px; }
  .model-r2 { display: block; font-size: 0.9rem; color: #6b7280; }
  .best-badge { position: absolute; top: 8px; right: 8px; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .info-tip { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9rem; color: #1e40af; }
  .layer-section { margin-bottom: 20px; padding: 16px; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb; }
  .layer-section h2 { margin: 0 0 12px 0; color: #1f2937; font-size: 1.1rem; }
  .input-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  label { display: flex; flex-direction: column; font-size: 0.8rem; color: #4b5563; font-weight: 500; white-space: nowrap; }
  input { margin-top: 4px; padding: 8px 6px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.9rem; transition: border-color 0.2s; width: 100%; box-sizing: border-box; }
  input:focus { outline: none; border-color: #8b5cf6; box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1); }
  .btn-row { display: flex; gap: 10px; margin-top: 24px; }
  .btn-primary, .btn-secondary, .btn-reset { flex: 1; padding: 14px 20px; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
  .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
  .btn-secondary { background: white; color: #8b5cf6; border: 2px solid #8b5cf6; }
  .btn-secondary:hover { background: #f3f0ff; transform: translateY(-2px); }
  .btn-reset { background: white; color: #64748b; border: 2px solid #cbd5e1; }
  .btn-reset:hover { background: #f8fafc; border-color: #94a3b8; transform: translateY(-2px); }
  .error { margin-top: 16px; padding: 12px; background: #fee2e2; border-left: 4px solid #ef4444; border-radius: 8px; color: #991b1b; font-weight: 500; }
  .results-section { height: 100%; display: flex; flex-direction: column; }
  .prediction-result { background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 28px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); }
  
  .gauge-container { 
    text-align: center; 
    margin-bottom: 28px; 
    padding: 20px;
    background: linear-gradient(135deg, #fafafa 0%, #ffffff 100%);
    border-radius: 12px;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
  }
  .gauge-svg { width: 100%; max-width: 320px; height: auto; margin: 0 auto; display: block; }
  .gauge-value { margin-top: 16px; }
  .gauge-number { 
    font-size: 3.5rem; 
    font-weight: 800; 
    line-height: 1; 
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .gauge-label { 
    font-size: 0.95rem; 
    color: #64748b; 
    font-weight: 600; 
    margin-top: 6px; 
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  
  .fos-status-box { 
    background: white; 
    padding: 16px; 
    border-radius: 8px; 
    margin-bottom: 16px; 
    text-align: center;
  }
  .status-text { 
    font-size: 1.1rem; 
    font-weight: 600; 
  }
  
  .fos-value { text-align: center; margin-bottom: 24px; }
  .fos-label { font-size: 1rem; color: #6b7280; font-weight: 600; margin-bottom: 8px; }
  .fos-number { font-size: 4rem; font-weight: 700; line-height: 1; margin-bottom: 8px; }
  .fos-status { font-size: 1.1rem; font-weight: 600; }
  .model-used { background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #e5e7eb; }
  .about-fos { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 20px; border-radius: 12px; border: 2px solid #10b981; margin-top: 20px; }
  .about-fos h3 { margin: 0 0 12px 0; color: #065f46; font-size: 1.1rem; }
  .about-fos p { margin: 0 0 12px 0; color: #047857; line-height: 1.6; }
  .about-fos ul { margin: 0; padding-left: 20px; color: #047857; }
  .about-fos li { margin-bottom: 8px; line-height: 1.6; }
  .no-prediction { text-align: center; padding: 60px 20px; color: #6b7280; }
  .placeholder-icon { font-size: 4rem; margin-bottom: 16px; opacity: 0.5; }
  .no-prediction h3 { color: #1f2937; margin: 0 0 12px 0; }
  .no-prediction p { margin: 0; line-height: 1.6; }
</style>
