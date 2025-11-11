# APPENDIX K5: LightGBM Algorithm

## Model Overview
**Light Gradient Boosting Machine (LightGBM)** is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It is designed for distributed and efficient training, particularly suited for large datasets. LightGBM uses a novel technique called Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to significantly speed up training while maintaining accuracy.

## Performance Metrics
- **R² Score (Training)**: 0.9872
- **RMSE**: 0.0407
- **MAE**: 0.0297
- **Rank**: 3rd among 6 models

## Model Configuration
```python
lgb.LGBMRegressor(
    n_estimators=200,      # Number of boosting iterations
    max_depth=10,          # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    random_state=42,
    verbose=-1
)
```

## Algorithm Principles

### 1. Leaf-wise Tree Growth
Unlike traditional level-wise growth, LightGBM grows trees leaf-wise (best-first):
- Chooses the leaf with maximum delta loss to grow
- More efficient than level-wise algorithms
- Can achieve better accuracy with same number of leaves

### 2. Gradient-based One-Side Sampling (GOSS)
LightGBM uses GOSS to reduce the number of data instances:
- Keeps instances with large gradients (large errors)
- Randomly samples instances with small gradients
- Maintains accuracy while reducing computation

### 3. Exclusive Feature Bundling (EFB)
Bundles mutually exclusive features together:
- Reduces number of features
- Decreases memory usage
- Speeds up training without losing information

## Mathematical Formulation

### Objective Function
$$L(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

Where:
- $l$ is the loss function (MSE for regression)
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is regularization
- $T$ is number of leaves
- $w_j$ is leaf weight

### Gradient and Hessian
For sample $i$:
$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$$

$$h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$

### Split Gain
For a split that divides instances into left ($I_L$) and right ($I_R$) sets:

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

### GOSS Algorithm
1. Sort instances by absolute value of gradients
2. Keep top $a \times 100\%$ instances with large gradients
3. Randomly sample $b \times 100\%$ from remaining instances
4. Amplify small gradient instances by constant $\frac{1-a}{b}$
5. Use this subset to estimate information gain

## Step-by-Step Training Process

### Step 1: Initialize Model
$$\hat{y}_i^{(0)} = \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$$

### Step 2: For each boosting iteration $t = 1$ to $T$:

**2.1. Calculate gradients and hessians**
```
For each sample i:
    g_i = ∂L/∂ŷ = 2(ŷ_i^(t-1) - y_i)
    h_i = ∂²L/∂ŷ² = 2
```

**2.2. Apply GOSS**
```
Sort samples by |g_i|
Keep top_a = a% samples with largest |g_i|
Randomly sample rand_b = b% from remaining samples
Amplify rand_b gradients by (1-a)/b
Combined_set = top_a ∪ rand_b
```

**2.3. Build tree using leaf-wise growth**
```
Initialize: root contains all samples in Combined_set
Priority_queue = [root]

While priority_queue not empty and num_leaves < max_leaves:
    leaf = priority_queue.pop_best()  # Leaf with highest potential gain
    
    best_split = find_best_split(leaf)
    if best_split.gain > 0:
        left_child, right_child = split(leaf, best_split)
        priority_queue.push(left_child)
        priority_queue.push(right_child)
```

**2.4. Find best split for a leaf**
```
For each feature f:
    Create histogram of gradients and hessians
    For each bin threshold:
        Calculate split gain using formula above
        Track best split
```

**2.5. Calculate leaf weights**
```
For each leaf j:
    w_j = -Σg_i / (Σh_i + λ)
    where sum is over samples in leaf j
```

**2.6. Update predictions**
```
For each sample i:
    ŷ_i^(t) = ŷ_i^(t-1) + η · w_leaf(i)
where:
    η = learning_rate = 0.1
    w_leaf(i) = weight of leaf containing sample i
```

### Step 3: Final Prediction
$$\hat{y}_i = \hat{y}_i^{(0)} + \sum_{t=1}^{T} \eta \cdot f_t(x_i)$$

## Pseudocode

```
Algorithm: LightGBM Training

Input: Training data D = {(x_i, y_i)}, i = 1 to n
       Number of iterations T = 200
       Max depth = 10
       Learning rate η = 0.1
       GOSS parameters a, b

Output: Ensemble model F(x)

1. Initialize:
   F_0(x) = mean(y)
   
2. For t = 1 to T:
   
   a) Compute gradients and hessians:
      For i = 1 to n:
         g_i = 2(F_{t-1}(x_i) - y_i)
         h_i = 2
   
   b) Apply GOSS:
      Sort samples by |g_i| descending
      A = top a% samples
      B = random b% from remaining
      Amplify B gradients by (1-a)/b
      D_sampled = A ∪ B
   
   c) Build tree leaf-wise:
      Initialize root with D_sampled
      Q = priority_queue([root])
      
      While |Q| > 0 and num_leaves < max_leaves:
         node = Q.pop_max()  # Node with max gain
         
         # Find best split
         best_gain = -∞
         For each feature f in node:
            Build histogram
            For each threshold in histogram:
               gain = calculate_split_gain(f, threshold)
               If gain > best_gain:
                  best_gain = gain
                  best_split = (f, threshold)
         
         If best_gain > 0:
            left, right = split(node, best_split)
            Q.push(left)
            Q.push(right)
      
      # Assign leaf weights
      For each leaf L:
         w_L = -Σ_{i∈L} g_i / (Σ_{i∈L} h_i + λ)
   
   d) Update model:
      F_t(x) = F_{t-1}(x) + η · tree_t(x)

3. Return F_T(x)
```

## Key Features

### 1. Speed Advantages
- **Histogram-based algorithm**: Bucketing continuous values into bins reduces complexity
- **Leaf-wise growth**: More efficient than level-wise
- **GOSS**: Reduces data size while maintaining accuracy
- **EFB**: Reduces feature dimensionality

### 2. Memory Efficiency
- Uses histograms instead of pre-sorted feature values
- Bundles exclusive features together
- Efficient sparse feature handling

### 3. Accuracy Features
- Leaf-wise growth can achieve better accuracy
- GOSS maintains information while sampling
- Supports categorical features natively

## Comparison with Traditional Gradient Boosting

| Aspect | LightGBM | Traditional GB |
|--------|----------|----------------|
| Tree Growth | Leaf-wise (best-first) | Level-wise |
| Data Sampling | GOSS (selective) | Full dataset |
| Feature Handling | Histogram + EFB | Pre-sorted values |
| Speed | Much faster | Slower |
| Memory | Lower | Higher |
| Overfitting Risk | Higher (leaf-wise) | Lower (level-wise) |

## Application to Slope Stability

### Input Features (4 averaged parameters):
1. **Cohesion** (c): 10-60 kPa
2. **Friction Angle** (φ): 20-40°
3. **Unit Weight** (γ): 16-22 kN/m³
4. **Ru** (pore pressure ratio): 0-0.5

### Training Process:
1. **200 trees** built sequentially using GOSS
2. Each tree focuses on samples with large prediction errors
3. **Leaf-wise growth** achieves depth up to 10 levels
4. **Learning rate 0.1** prevents overfitting
5. Predictions combined: $\text{FoS} = \bar{y} + 0.1 \sum_{t=1}^{200} f_t(\mathbf{x})$

### Performance Analysis:
- **R² = 0.9872**: Excellent fit to training data
- **RMSE = 0.0407**: Low average prediction error
- **MAE = 0.0297**: Typical error ~0.03 in FoS units
- **Speed**: Fastest training among all models
- **Rank**: 3rd best overall performance

## Advantages for FoS Prediction

1. **Speed**: Trains much faster than other ensemble methods
2. **Memory Efficient**: Can handle large datasets easily
3. **Accuracy**: Very high R² score (98.72%)
4. **Robustness**: GOSS reduces noise in training data
5. **Handles Non-linearity**: Captures complex relationships in geotechnical data

## Limitations

1. **Overfitting Risk**: Leaf-wise growth can overfit on small datasets
2. **Hyperparameter Sensitivity**: Requires careful tuning of max_depth
3. **Less Interpretable**: Complex ensemble structure
4. **Stability**: Small changes in data can affect leaf-wise splits

## Complexity Analysis

- **Training Time**: $O(n \times m \times k \times \log n)$ where:
  - $n$ = number of samples (reduced by GOSS)
  - $m$ = number of features (reduced by EFB)
  - $k$ = number of bins in histogram
  - $\log n$ for tree depth

- **Prediction Time**: $O(T \times \text{tree\_depth})$
  - $T = 200$ trees
  - Average depth ≈ 10

- **Memory**: $O(T \times L)$ where $L$ = average leaves per tree

## Conclusion

LightGBM achieves excellent performance (3rd place, R² = 0.9872) with significantly faster training time compared to traditional gradient boosting. Its innovative GOSS and EFB techniques make it ideal for slope stability prediction where:
- Training speed is important
- Dataset size may grow over time
- High accuracy is required
- Memory efficiency matters

The model successfully captures the complex non-linear relationships between soil parameters and Factor of Safety while maintaining computational efficiency.
