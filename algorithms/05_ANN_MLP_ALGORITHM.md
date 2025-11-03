# ALGORITHM USED FOR ARTIFICIAL NEURAL NETWORK (MLP REGRESSOR)

## Algorithm: Multi-Layer Perceptron with Backpropagation

### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

### Step 2: Feature Scaling
**Critical for neural networks:**
1. Standardize features using StandardScaler:
   ```
   X_scaled = (X - μ) / σ
   
   Where:
   μ = mean of feature
   σ = standard deviation of feature
   ```
2. Ensures all features have similar scale (mean=0, std=1)
3. Prevents features with large magnitudes from dominating

### Step 3: Initialize Network Architecture
1. **Define layer structure:**
   - Input layer: 30 neurons (one per feature)
   - Hidden layers: (100, 50) neurons
     - First hidden: 100 neurons
     - Second hidden: 50 neurons
   - Output layer: 1 neuron (FoS prediction)

2. **Network topology:**
   ```
   Input (30) → Hidden1 (100) → Hidden2 (50) → Output (1)
   ```

### Step 4: Initialize Weights and Biases
For each layer l:

1. **Weight initialization (Xavier/Glorot):**
   ```
   W⁽ˡ⁾ ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))
   
   Where:
   nᵢₙ = number of input neurons
   nₒᵤₜ = number of output neurons
   ```

2. **Bias initialization:**
   ```
   b⁽ˡ⁾ = 0 (initialize to zeros)
   ```

3. **Parameter dimensions:**
   - W⁽¹⁾: 30 × 100, b⁽¹⁾: 100
   - W⁽²⁾: 100 × 50, b⁽²⁾: 50
   - W⁽³⁾: 50 × 1, b⁽³⁾: 1

### Step 5: Set Hyperparameters
1. Activation function: activation = 'relu'
2. Solver: solver = 'adam' (Adaptive Moment Estimation)
3. Learning rate: learning_rate_init = 0.001
4. Max iterations: max_iter = 1000
5. Random state: random_state = 42
6. Batch size: Default (min(200, n_samples))
7. Momentum: beta_1 = 0.9, beta_2 = 0.999 (for Adam)
8. Epsilon: epsilon = 1e-08 (for numerical stability)

### Step 6: Training Phase - Forward Propagation

For each training sample (or mini-batch):

#### Step 6.1: Forward Pass Through Network

**Input Layer:**
```
a⁽⁰⁾ = X (scaled input features)
```

**For each hidden layer l = 1, 2:**

1. **Linear transformation:**
   ```
   z⁽ˡ⁾ = W⁽ˡ⁾ᵀ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
   ```

2. **Apply ReLU activation:**
   ```
   a⁽ˡ⁾ = ReLU(z⁽ˡ⁾) = max(0, z⁽ˡ⁾)
   
   ReLU(x) = {
     x,  if x > 0
     0,  if x ≤ 0
   }
   ```

**Output Layer (l = 3):**

1. **Linear transformation:**
   ```
   z⁽³⁾ = W⁽³⁾ᵀ · a⁽²⁾ + b⁽³⁾
   ```

2. **Identity activation (for regression):**
   ```
   ŷ = a⁽³⁾ = z⁽³⁾
   ```

### Step 7: Compute Loss Function
1. **Mean Squared Error (MSE):**
   ```
   L = (1/n)·Σᵢ (yᵢ - ŷᵢ)²
   ```

2. **For single sample:**
   ```
   L = (y - ŷ)²
   ```

### Step 8: Backward Propagation (Backprop)

Compute gradients layer by layer, from output to input:

#### Step 8.1: Output Layer Gradient
```
δ⁽³⁾ = ∂L/∂z⁽³⁾ = 2(ŷ - y)

∂L/∂W⁽³⁾ = a⁽²⁾ · δ⁽³⁾
∂L/∂b⁽³⁾ = δ⁽³⁾
```

#### Step 8.2: Hidden Layer 2 Gradient
```
δ⁽²⁾ = (W⁽³⁾ · δ⁽³⁾) ⊙ ReLU'(z⁽²⁾)

Where ReLU'(z) = {
  1,  if z > 0
  0,  if z ≤ 0
}

∂L/∂W⁽²⁾ = a⁽¹⁾ · δ⁽²⁾
∂L/∂b⁽²⁾ = δ⁽²⁾
```

#### Step 8.3: Hidden Layer 1 Gradient
```
δ⁽¹⁾ = (W⁽²⁾ · δ⁽²⁾) ⊙ ReLU'(z⁽¹⁾)

∂L/∂W⁽¹⁾ = a⁽⁰⁾ · δ⁽¹⁾
∂L/∂b⁽¹⁾ = δ⁽¹⁾
```

### Step 9: Update Weights Using Adam Optimizer

For each parameter θ (weights and biases):

#### Step 9.1: Compute Gradient
```
g_t = ∂L/∂θ
```

#### Step 9.2: Update Biased First Moment Estimate
```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

Where β₁ = 0.9
```

#### Step 9.3: Update Biased Second Moment Estimate
```
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

Where β₂ = 0.999
```

#### Step 9.4: Compute Bias-Corrected Moment Estimates
```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

Where t = current iteration number
```

#### Step 9.5: Update Parameters
```
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

Where:
α = learning_rate_init = 0.001
ε = 1e-08 (for numerical stability)
```

### Step 10: Repeat Training
1. **For each epoch (up to max_iter = 1000):**
   - Shuffle training data
   - Divide into mini-batches
   - For each mini-batch:
     - Forward propagation (Step 6)
     - Compute loss (Step 7)
     - Backward propagation (Step 8)
     - Update weights with Adam (Step 9)

2. **Early stopping criteria:**
   - If validation loss doesn't improve for n_iter_no_change iterations
   - Or max_iter reached
   - Or loss < tolerance

### Step 11: Prediction Phase
For each new sample x:

1. **Scale features using fitted scaler:**
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Forward propagation:**
   ```
   z⁽¹⁾ = W⁽¹⁾ᵀ · x_scaled + b⁽¹⁾
   a⁽¹⁾ = ReLU(z⁽¹⁾)
   
   z⁽²⁾ = W⁽²⁾ᵀ · a⁽¹⁾ + b⁽²⁾
   a⁽²⁾ = ReLU(z⁽²⁾)
   
   z⁽³⁾ = W⁽³⁾ᵀ · a⁽²⁾ + b⁽³⁾
   ŷ = z⁽³⁾
   ```

3. Return prediction ŷ

### Step 12: Model Evaluation
1. Make predictions on test set
2. Calculate R² Score:
   ```
   R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   ```
3. Calculate RMSE:
   ```
   RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
   ```
4. Calculate MAE:
   ```
   MAE = Σ|yᵢ - ŷᵢ| / n
   ```

### Step 13: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train MLP on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score and standard deviation

### Step 14: Save Model and Scaler
1. Serialize trained MLP model using joblib
2. Save StandardScaler (critical for predictions!)
3. Save to 'ann_mlp.joblib'
4. Store performance metrics in JSON

### Step 15: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_layer_sizes | (100, 50) | Two hidden layers |
| activation | 'relu' | ReLU activation function |
| solver | 'adam' | Adam optimizer |
| learning_rate_init | 0.001 | Initial learning rate |
| max_iter | 1000 | Maximum iterations |
| random_state | 42 | Random seed |
| beta_1 | 0.9 | Adam momentum parameter |
| beta_2 | 0.999 | Adam second moment parameter |
| epsilon | 1e-08 | Numerical stability |

## Network Architecture

```
Layer          Neurons    Weights        Biases    Activation
─────────────────────────────────────────────────────────────
Input          30         -              -         -
Hidden 1       100        30 × 100       100       ReLU
Hidden 2       50         100 × 50       50        ReLU
Output         1          50 × 1         1         Identity
─────────────────────────────────────────────────────────────
Total Parameters: 8,251 (3,000 + 10,100 + 5,050 + 51)
```

## Performance Metrics

- **R² Score**: 0.8916 (89.16% variance explained)
- **RMSE**: 0.0662
- **MAE**: 0.0473
- **Status**: Fifth best performing model
- **Training Time**: Moderate (iterative optimization)

## Advantages
- **Universal approximation** - Can learn any continuous function
- **Non-linear modeling** - Captures complex relationships
- **Feature learning** - Automatically learns representations
- **Flexible architecture** - Can add/remove layers
- **Parallel computation** - Matrix operations are GPU-friendly

## Disadvantages
- **Requires feature scaling** - Sensitive to input magnitudes
- **Hyperparameter tuning** - Architecture, learning rate, etc.
- **Black box nature** - Hard to interpret
- **Risk of overfitting** - Needs regularization/dropout
- **Longer training time** - Iterative optimization

## Mathematical Formulation

**Network Function:**
```
ŷ = f(x; W, b) = W⁽³⁾ᵀ · ReLU(W⁽²⁾ᵀ · ReLU(W⁽¹⁾ᵀ · x + b⁽¹⁾) + b⁽²⁾) + b⁽³⁾
```

**Loss Function:**
```
L(W, b) = (1/n)·Σᵢ₌₁ⁿ (yᵢ - f(xᵢ; W, b))²
```

**Backpropagation Chain Rule:**
```
∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
         = δ⁽ˡ⁾ · a⁽ˡ⁻¹⁾ᵀ
```

**Adam Update Rule:**
```
m_t = β₁·m_{t-1} + (1-β₁)·∇L
v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²

θ_t = θ_{t-1} - α · m̂_t/(√v̂_t + ε)

Where:
m̂_t = m_t/(1-β₁ᵗ)  (bias-corrected first moment)
v̂_t = v_t/(1-β₂ᵗ)  (bias-corrected second moment)
```

**ReLU Properties:**
```
ReLU(x) = max(0, x)

Advantages:
- Solves vanishing gradient problem
- Sparse activation (only ~50% neurons active)
- Computationally efficient
- Allows faster convergence

Derivative:
ReLU'(x) = 1 if x > 0, else 0
```

**Xavier Initialization Rationale:**
```
Var(W) = 2/(nᵢₙ + nₒᵤₜ)

Maintains variance of activations across layers
Prevents exploding/vanishing gradients during initialization
```
