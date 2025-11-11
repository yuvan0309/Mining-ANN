# APPENDIX K6: Artificial Neural Network (ANN) Algorithm

## Model Overview
**Artificial Neural Network (Multi-Layer Perceptron)** is a feedforward neural network that learns complex non-linear relationships through multiple layers of interconnected neurons. The model uses backpropagation with the Adam optimizer to iteratively adjust weights and minimize prediction error. It is particularly effective for capturing intricate patterns in geotechnical data.

## Performance Metrics
- **R² Score (Training)**: 0.9316
- **RMSE**: 0.0940
- **MAE**: 0.0694
- **Rank**: 6th among 6 models

## Model Configuration
```python
MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
    activation='relu',                  # ReLU activation function
    solver='adam',                      # Adam optimizer
    max_iter=1000,                      # Maximum iterations
    random_state=42,
    early_stopping=True                 # Prevent overfitting
)
```

## Network Architecture

### Layer Structure
```
Input Layer:    4 neurons (Cohesion, Friction Angle, Unit Weight, Ru)
                    ↓
Hidden Layer 1: 100 neurons (ReLU activation)
                    ↓
Hidden Layer 2: 50 neurons (ReLU activation)
                    ↓
Hidden Layer 3: 25 neurons (ReLU activation)
                    ↓
Output Layer:   1 neuron (Factor of Safety)
```

### Total Parameters
- **Input → Hidden1**: $4 \times 100 + 100 = 500$ (weights + biases)
- **Hidden1 → Hidden2**: $100 \times 50 + 50 = 5,050$
- **Hidden2 → Hidden3**: $50 \times 25 + 25 = 1,275$
- **Hidden3 → Output**: $25 \times 1 + 1 = 26$
- **Total**: $500 + 5,050 + 1,275 + 26 = 6,851$ parameters

## Mathematical Formulation

### Forward Propagation

**Input Layer (Layer 0)**:
$$\mathbf{a}^{(0)} = \mathbf{x} = [x_1, x_2, x_3, x_4]^T$$

Where:
- $x_1$ = Cohesion
- $x_2$ = Friction Angle
- $x_3$ = Unit Weight
- $x_4$ = Ru

**Hidden Layer 1 (100 neurons)**:
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{a}^{(0)} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) = \max(0, \mathbf{z}^{(1)})$$

**Hidden Layer 2 (50 neurons)**:
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$
$$\mathbf{a}^{(2)} = \text{ReLU}(\mathbf{z}^{(2)}) = \max(0, \mathbf{z}^{(2)})$$

**Hidden Layer 3 (25 neurons)**:
$$\mathbf{z}^{(3)} = \mathbf{W}^{(3)} \mathbf{a}^{(2)} + \mathbf{b}^{(3)}$$
$$\mathbf{a}^{(3)} = \text{ReLU}(\mathbf{z}^{(3)}) = \max(0, \mathbf{z}^{(3)})$$

**Output Layer (1 neuron)**:
$$\mathbf{z}^{(4)} = \mathbf{W}^{(4)} \mathbf{a}^{(3)} + \mathbf{b}^{(4)}$$
$$\hat{y} = \mathbf{a}^{(4)} = \mathbf{z}^{(4)}$$ (linear activation for regression)

### Loss Function
Mean Squared Error (MSE):
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Activation Functions

**ReLU (Rectified Linear Unit)**:
$$\text{ReLU}(z) = \begin{cases} 
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases}$$

**ReLU Derivative**:
$$\text{ReLU}'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases}$$

### Backpropagation

**Output Layer Gradient**:
$$\delta^{(4)} = \frac{\partial L}{\partial \mathbf{z}^{(4)}} = 2(\hat{y} - y)$$

**Hidden Layer 3 Gradient**:
$$\delta^{(3)} = (\mathbf{W}^{(4)})^T \delta^{(4)} \odot \text{ReLU}'(\mathbf{z}^{(3)})$$

**Hidden Layer 2 Gradient**:
$$\delta^{(2)} = (\mathbf{W}^{(3)})^T \delta^{(3)} \odot \text{ReLU}'(\mathbf{z}^{(2)})$$

**Hidden Layer 1 Gradient**:
$$\delta^{(1)} = (\mathbf{W}^{(2)})^T \delta^{(2)} \odot \text{ReLU}'(\mathbf{z}^{(1)})$$

Where $\odot$ denotes element-wise multiplication.

### Weight Updates (Adam Optimizer)

**Gradient Calculation**:
$$\mathbf{g}_t^{(l)} = \frac{\partial L}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$$

**First Moment Estimate (mean)**:
$$\mathbf{m}_t^{(l)} = \beta_1 \mathbf{m}_{t-1}^{(l)} + (1 - \beta_1) \mathbf{g}_t^{(l)}$$

**Second Moment Estimate (variance)**:
$$\mathbf{v}_t^{(l)} = \beta_2 \mathbf{v}_{t-1}^{(l)} + (1 - \beta_2) (\mathbf{g}_t^{(l)})^2$$

**Bias Correction**:
$$\hat{\mathbf{m}}_t^{(l)} = \frac{\mathbf{m}_t^{(l)}}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}}_t^{(l)} = \frac{\mathbf{v}_t^{(l)}}{1 - \beta_2^t}$$

**Weight Update**:
$$\mathbf{W}_t^{(l)} = \mathbf{W}_{t-1}^{(l)} - \alpha \frac{\hat{\mathbf{m}}_t^{(l)}}{\sqrt{\hat{\mathbf{v}}_t^{(l)}} + \epsilon}$$

Where:
- $\alpha$ = learning rate (adaptive)
- $\beta_1 = 0.9$ (first moment decay)
- $\beta_2 = 0.999$ (second moment decay)
- $\epsilon = 10^{-8}$ (numerical stability)

## Step-by-Step Training Process

### Step 1: Initialize Network
```
For each layer l:
    Initialize weights W^(l) using Xavier/He initialization:
        W^(l) ~ Normal(0, sqrt(2/n_in))
    Initialize biases b^(l) = 0
    Initialize Adam parameters:
        m^(l) = 0  (first moment)
        v^(l) = 0  (second moment)
```

### Step 2: For each epoch (max 1000 iterations):

**2.1. Forward Pass**
```
For each sample i in training set:
    a^(0) = x_i (input features)
    
    # Hidden Layer 1
    z^(1) = W^(1) · a^(0) + b^(1)
    a^(1) = ReLU(z^(1))
    
    # Hidden Layer 2
    z^(2) = W^(2) · a^(1) + b^(2)
    a^(2) = ReLU(z^(2))
    
    # Hidden Layer 3
    z^(3) = W^(3) · a^(2) + b^(3)
    a^(3) = ReLU(z^(3))
    
    # Output Layer
    z^(4) = W^(4) · a^(3) + b^(4)
    ŷ_i = z^(4)
    
    # Calculate loss
    loss_i = (y_i - ŷ_i)²
```

**2.2. Backward Pass**
```
# Output layer gradient
δ^(4) = 2(ŷ_i - y_i)

# Hidden layer 3 gradient
δ^(3) = (W^(4))^T · δ^(4) ⊙ ReLU'(z^(3))

# Hidden layer 2 gradient
δ^(2) = (W^(3))^T · δ^(3) ⊙ ReLU'(z^(2))

# Hidden layer 1 gradient
δ^(1) = (W^(2))^T · δ^(2) ⊙ ReLU'(z^(1))
```

**2.3. Compute Gradients**
```
For each layer l:
    ∂L/∂W^(l) = δ^(l) · (a^(l-1))^T
    ∂L/∂b^(l) = δ^(l)
```

**2.4. Update Weights using Adam**
```
t = iteration counter

For each layer l:
    # Compute biased moments
    m^(l) = β₁ · m^(l) + (1-β₁) · ∂L/∂W^(l)
    v^(l) = β₂ · v^(l) + (1-β₂) · (∂L/∂W^(l))²
    
    # Bias correction
    m̂^(l) = m^(l) / (1 - β₁^t)
    v̂^(l) = v^(l) / (1 - β₂^t)
    
    # Update weights
    W^(l) = W^(l) - α · m̂^(l) / (√v̂^(l) + ε)
    
    # Same for biases
    b^(l) = b^(l) - α · m̂_b^(l) / (√v̂_b^(l) + ε)
```

**2.5. Early Stopping Check**
```
If validation loss hasn't improved for 10 epochs:
    Stop training
    Restore best weights
```

### Step 3: Final Prediction
$$\hat{y} = f_{\theta}(\mathbf{x}) = \text{forward\_pass}(\mathbf{x}, \mathbf{W}^*, \mathbf{b}^*)$$

## Pseudocode

```
Algorithm: Multi-Layer Perceptron Training

Input: Training data D = {(x_i, y_i)}, i = 1 to n
       Architecture: [4, 100, 50, 25, 1]
       Max iterations = 1000
       Early stopping patience = 10

Output: Trained network parameters {W^(l), b^(l)}

1. Initialize:
   For l = 1 to 4:
      W^(l) ~ Normal(0, sqrt(2/n_in[l]))
      b^(l) = 0
      m^(l) = 0, v^(l) = 0
   
   best_loss = ∞
   patience_counter = 0
   t = 0  # iteration counter

2. For epoch = 1 to 1000:
   
   # Shuffle training data
   Shuffle(D)
   
   # Mini-batch training
   For each mini-batch B in D:
      t = t + 1
      
      For each sample (x, y) in B:
         # Forward pass
         a^(0) = x
         For l = 1 to 4:
            z^(l) = W^(l) · a^(l-1) + b^(l)
            If l < 4:
               a^(l) = ReLU(z^(l))
            Else:
               a^(l) = z^(l)  # Linear output
         
         ŷ = a^(4)
         loss = (y - ŷ)²
         
         # Backward pass
         δ^(4) = 2(ŷ - y)
         For l = 3 down to 1:
            δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ ReLU'(z^(l))
         
         # Accumulate gradients
         For l = 1 to 4:
            g_W^(l) += δ^(l) · (a^(l-1))^T
            g_b^(l) += δ^(l)
      
      # Adam update
      For l = 1 to 4:
         # Update moments
         m^(l) = β₁ · m^(l) + (1-β₁) · g_W^(l)
         v^(l) = β₂ · v^(l) + (1-β₂) · (g_W^(l))²
         
         # Bias correction
         m̂^(l) = m^(l) / (1 - β₁^t)
         v̂^(l) = v^(l) / (1 - β₂^t)
         
         # Update parameters
         W^(l) -= α · m̂^(l) / (√v̂^(l) + ε)
         b^(l) -= α · m̂_b^(l) / (√v̂_b^(l) + ε)
   
   # Validate
   val_loss = compute_validation_loss()
   
   If val_loss < best_loss:
      best_loss = val_loss
      save_weights()
      patience_counter = 0
   Else:
      patience_counter += 1
   
   If patience_counter >= 10:
      restore_best_weights()
      Break  # Early stopping

3. Return {W^(l), b^(l)} for l = 1 to 4
```

## Key Features

### 1. Deep Architecture
- **3 hidden layers**: Progressively reduces dimensionality (100 → 50 → 25)
- **Hierarchical learning**: Each layer learns increasingly abstract features
- **6,851 parameters**: Sufficient capacity for complex patterns

### 2. ReLU Activation
- **Non-linearity**: Captures complex relationships
- **Gradient flow**: Mitigates vanishing gradient problem
- **Sparse activation**: Many neurons output zero, improving efficiency

### 3. Adam Optimizer
- **Adaptive learning rates**: Different rates for each parameter
- **Momentum**: Accelerates convergence
- **Variance scaling**: Handles noisy gradients well

### 4. Early Stopping
- **Prevents overfitting**: Stops when validation loss plateaus
- **Automatic**: No need to guess optimal epochs
- **Best model selection**: Restores weights from best epoch

## Application to Slope Stability

### Input Features (4 averaged parameters):
1. **Cohesion** (c): 10-60 kPa → Standardized
2. **Friction Angle** (φ): 20-40° → Standardized
3. **Unit Weight** (γ): 16-22 kN/m³ → Standardized
4. **Ru** (pore pressure ratio): 0-0.5 → Standardized

### Feature Standardization:
$$x_{std} = \frac{x - \mu}{\sigma}$$

Applied before feeding into network for stable training.

### Network Processing:
1. **Input standardized** using training set statistics
2. **Hidden Layer 1**: Extracts 100 low-level features
3. **Hidden Layer 2**: Combines into 50 mid-level features
4. **Hidden Layer 3**: Produces 25 high-level representations
5. **Output**: Single FoS prediction

### Training Process:
- **Maximum 1000 epochs** with early stopping
- **Adam optimizer** with adaptive learning rates
- **Batch processing** for efficiency
- **Validation monitoring** prevents overfitting

### Performance Analysis:
- **R² = 0.9316**: Good fit, but lowest among 6 models
- **RMSE = 0.0940**: Higher error than tree-based models
- **MAE = 0.0694**: Typical error ~0.07 in FoS units
- **Training time**: Moderate (longer than simple models)

## Advantages for FoS Prediction

1. **Universal Approximator**: Can theoretically approximate any continuous function
2. **Non-linear Relationships**: Captures complex interactions automatically
3. **Feature Learning**: Automatically discovers relevant patterns
4. **Flexible Architecture**: Can be scaled for more complex problems
5. **Well-established**: Mature theory and optimization techniques

## Limitations

1. **Requires More Data**: 6,851 parameters need substantial training data
2. **Black Box**: Difficult to interpret learned features
3. **Hyperparameter Sensitivity**: Architecture and learning rate require tuning
4. **Training Time**: Slower convergence than tree-based methods
5. **Local Minima**: May get stuck in suboptimal solutions
6. **Overfitting Risk**: Large parameter space can memorize training data

## Comparison with Tree-Based Models

| Aspect | ANN | Tree-Based (GB, RF, XGB) |
|--------|-----|--------------------------|
| R² Score | 0.9316 (lowest) | 0.95-0.99 (higher) |
| Training Speed | Moderate | Faster |
| Interpretability | Very low | Low to moderate |
| Overfitting Risk | High | Moderate |
| Feature Engineering | Automatic | Manual feature importance |
| Extrapolation | Poor | Poor |

## Complexity Analysis

- **Training Time**: $O(E \times n \times m \times P)$ where:
  - $E$ = number of epochs (≤1000)
  - $n$ = number of samples
  - $m$ = mini-batch size
  - $P = 6,851$ = total parameters

- **Forward Pass**: $O(P)$ operations
  - $4 \times 100 + 100 \times 50 + 50 \times 25 + 25 \times 1 = 6,425$ multiplications

- **Backward Pass**: $O(P)$ operations (same as forward)

- **Memory**: $O(P + n \times \text{batch\_size})$

## Conclusion

While the ANN achieves respectable performance (R² = 0.9316), it ranks 6th among the 6 models tested. For slope stability prediction with limited data (288 training samples), tree-based ensemble methods outperform the neural network approach. The ANN's 6,851 parameters may be excessive for this dataset size, leading to:

- **Underfitting relative to potential**: Not enough data to fully utilize network capacity
- **Slower convergence**: More iterations needed compared to tree-based models
- **Higher error**: RMSE=0.094 vs 0.024 for Gradient Boosting

However, ANNs remain valuable for:
- **Larger datasets**: Performance would improve with more training data
- **Feature discovery**: Automatic learning of complex patterns
- **Transfer learning**: Pre-trained networks could be fine-tuned
- **Real-time prediction**: Fast inference once trained

For production slope stability prediction with the current dataset, tree-based ensemble methods (Gradient Boosting, Random Forest, LightGBM, XGBoost) are recommended over ANN due to superior accuracy and efficiency.
