<script>
  import PredictionForm from './components/PredictionForm.svelte';
  import ResultsDisplay from './components/ResultsDisplay.svelte';
  import ModelInfo from './components/ModelInfo.svelte';
  
  let predictionResult = null;
  let isLoading = false;
  let selectedModel = 'gradient_boosting';
  
  function handlePrediction(event) {
    predictionResult = event.detail;
  }
  
  function handleLoading(event) {
    isLoading = event.detail;
  }
  
  function handleModelChange(event) {
    selectedModel = event.detail;
  }
</script>

<main class="app-container">
  <header class="app-header">
    <div class="header-content">
      <h1>⛰️ Factor of Safety Prediction System</h1>
      <p class="subtitle">Bishop's Simplified Method with Ru (Pore Pressure Ratio)</p>
    </div>
  </header>
  
  <div class="content-wrapper">
    <div class="main-content">
      <section class="prediction-section">
        <PredictionForm 
          bind:selectedModel={selectedModel}
          on:prediction={handlePrediction}
          on:loading={handleLoading}
          on:modelChange={handleModelChange}
        />
      </section>
      
      {#if predictionResult}
        <section class="results-section">
          <ResultsDisplay result={predictionResult} />
        </section>
      {/if}
    </div>
    
    <aside class="sidebar">
      <ModelInfo model={selectedModel} />
    </aside>
  </div>
  
  <footer class="app-footer">
    <p>Mining ANN Project | Factor of Safety Prediction | November 2025</p>
  </footer>
</main>

<style lang="css">
  .app-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  
  .app-header {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: white;
    padding: 2rem 0;
    box-shadow: var(--shadow-lg);
  }
  
  .header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  .app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  
  .subtitle {
    font-size: 1rem;
    opacity: 0.9;
    font-weight: 400;
  }
  
  .content-wrapper {
    flex: 1;
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
    width: 100%;
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 2rem;
  }
  
  .main-content {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }
  
  .prediction-section,
  .results-section {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color);
  }
  
  .sidebar {
    position: sticky;
    top: 2rem;
    height: fit-content;
  }
  
  .app-footer {
    background: var(--text-primary);
    color: white;
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
  }
  
  .app-footer p {
    font-size: 0.875rem;
    opacity: 0.8;
  }
  
  @media (max-width: 968px) {
    .content-wrapper {
      grid-template-columns: 1fr;
    }
    
    .sidebar {
      position: static;
    }
    
    .app-header h1 {
      font-size: 1.5rem;
    }
  }
</style>
