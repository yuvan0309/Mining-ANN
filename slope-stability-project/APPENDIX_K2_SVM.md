# APPENDIX K2

## ALGORITHM USED FOR SUPPORT VECTOR MACHINE (SVM) REGRESSION

### Algorithm: Support Vector Regression (SVR) with RBF Kernel

---

## Step 1: Load and Prepare Data

1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

---

## Step 2: Initialize Hyperparameters

1. **Set kernel type**: RBF (Radial Basis Function)
   - RBF kernel: K(x, x') = exp(-γ||x - x'||²)

2. **Initialize regularization parameter**: C = 1.0
   - Controls trade-off between model complexity and training error

3. **Initialize kernel coefficient**: γ = 'scale'
   - γ = 1 / (n_features × variance(X))

4. **Set epsilon**: ε = 0.1
   - Defines epsilon-tube for loss function

---

## Step 3: Feature Scaling

1. Create StandardScaler transformer
2. Fit scaler on training data:
   - μ (mean) = Σx / n
   - σ (std) = √(Σ(x - μ)² / n)
3. Transform features:
   - X_scaled = (X - μ) / σ
4. Apply same transformation to test data

---

## Step 4: Build SVM Model

1. Initialize SVR estimator with parameters:
   - kernel = 'rbf'
   - C = 1.0
   - gamma = 'scale'
   - epsilon = 0.1
2. Create pipeline combining scaler and SVR

---

## Step 5: Training Phase

**For each training sample (xᵢ, yᵢ):**

1. **Compute kernel matrix K:**
   ```
   K[i,j] = exp(-γ||xᵢ - xⱼ||²)
   
   Where:
   γ = 1 / (n_features × var(X)) = 1 / (4 × var(X))
   ```

2. **Solve optimization problem:**
   
   **Minimize:**
   ```
   ½||w||² + C·Σᵢ(ξᵢ + ξᵢ*)
   ```
   
   **Subject to:**
   ```
   yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
   (w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
   ξᵢ, ξᵢ* ≥ 0
   ```
   
   **Where:**
   - w = weight vector
   - φ(x) = feature mapping
   - b = bias term
   - ξ = slack variables
   - ε = epsilon (0.1)
   - C = penalty parameter (1.0)

3. **Dual Formulation:**
   
   **Maximize:**
   ```
   L(α, α*) = Σᵢyᵢ(αᵢ - αᵢ*) - ε·Σᵢ(αᵢ + αᵢ*) 
              - ½·ΣᵢΣⱼ(αᵢ - αᵢ*)(αⱼ - αⱼ*)K(xᵢ, xⱼ)
   ```
   
   **Subject to:**
   ```
   Σᵢ(αᵢ - αᵢ*) = 0
   0 ≤ αᵢ, αᵢ* ≤ C
   ```

4. **Find support vectors:**
   - Identify samples with 0 < αᵢ < C
   - These become support vectors
   - Typically 30-50% of training samples

5. **Compute decision function:**
   ```
   f(x) = Σᵢ(αᵢ - αᵢ*)K(xᵢ, x) + b
   
   Where the sum is only over support vectors
   ```

6. **Calculate bias term b:**
   ```
   For any support vector xⱼ with 0 < αⱼ < C:
   b = yⱼ - Σᵢ(αᵢ - αᵢ*)K(xᵢ, xⱼ) - ε·sign(αⱼ - αⱼ*)
   ```

---

## Step 6: Prediction Phase

**For each new sample x:**

1. **Scale input features:**
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Compute kernel with all support vectors:**
   ```
   K(xⱼ, x) = exp(-γ||xⱼ - x||²)
   
   Where:
   - xⱼ = support vectors
   - γ = 1/(n_features × var(X))
   ```

3. **Calculate prediction:**
   ```
   ŷ = Σⱼ(αⱼ - αⱼ*)K(xⱼ, x) + b
   
   Only sum over support vectors (not all training samples)
   ```

4. **Return predicted FoS value**

---

## Step 7: Model Evaluation

1. Make predictions on test set
2. Calculate R² Score:
   ```
   R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   R² = 0.9498 (94.98%)
   ```

3. Calculate RMSE:
   ```
   RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
   RMSE = 0.0451
   ```

4. Calculate MAE:
   ```
   MAE = Σ|yᵢ - ŷᵢ| / n
   MAE = 0.0325
   ```

---

## Step 8: Cross-Validation

1. Split data into k=5 folds
2. For each fold i:
   - Train on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score

---

## Step 9: Support Vector Analysis

1. **Count support vectors:**
   - Total: ~120-150 out of 288 samples (40-50%)
   
2. **Categorize support vectors:**
   - Margin support vectors: 0 < α < C
   - Error support vectors: α = C
   
3. **Analyze support vector distribution:**
   - Per class/region analysis
   - Feature space visualization

---

## Step 10: Save Model

1. Serialize trained pipeline using joblib
2. Save to 'best_model_svm.pkl'
3. Store performance metrics in JSON
4. Save support vector indices

---

## Step 11: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| kernel | 'rbf' | Radial Basis Function kernel |
| C | 1.0 | Regularization parameter |
| gamma | 'scale' | Kernel coefficient (auto-calculated) |
| epsilon | 0.1 | Epsilon-tube width |
| cache_size | 200 | Kernel cache size (MB) |
| max_iter | -1 | No iteration limit |

---

## Performance Metrics

- **R² Score (Test)**: 0.9498 (94.98% variance explained)
- **RMSE**: 0.0451
- **MAE**: 0.0325
- **Number of Support Vectors**: ~40-50% of training data
- **Status**: Second best performing model
- **Training Time**: Moderate (kernel computation intensive)

---

## Advantages

1. **Effective in high-dimensional spaces**: Works well with 4 features
2. **Memory efficient**: Uses only support vectors (not all training data)
3. **Robust to outliers**: Within epsilon-tube
4. **Non-linear modeling**: RBF kernel captures complex relationships
5. **Generalization**: Strong performance on unseen data
6. **Versatile**: Can handle different kernel types

---

## Disadvantages

1. **Kernel selection**: Requires choosing appropriate kernel
2. **Hyperparameter tuning**: Sensitive to C, γ, and ε values
3. **Computational cost**: O(n²) to O(n³) for kernel computation
4. **No probabilistic output**: Does not provide confidence intervals directly
5. **Black box**: Less interpretable than tree-based models

---

## Mathematical Formulation

### Decision Function:
```
f(x) = Σᵢ₌₁ⁿ(αᵢ - αᵢ*)K(xᵢ, x) + b

Where:
- αᵢ, αᵢ* = Lagrange multipliers
- K(xᵢ, x) = RBF kernel function
- b = bias term
- Sum over support vectors only
```

### RBF Kernel:
```
K(x, x') = exp(-γ||x - x'||²)

Where:
- γ = 1/(n_features × var(X))
- ||x - x'||² = Euclidean distance squared
```

### Loss Function (ε-insensitive):
```
L_ε(y, f(x)) = {
    0,                    if |y - f(x)| ≤ ε
    |y - f(x)| - ε,      otherwise
}
```

### Primal Optimization Problem:
```
min  ½||w||² + C·Σᵢ(ξᵢ + ξᵢ*)
w,b,ξ

Subject to:
yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
(w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
ξᵢ, ξᵢ* ≥ 0
```

### Dual Optimization Problem:
```
max  Σᵢyᵢ(αᵢ - αᵢ*) - ε·Σᵢ(αᵢ + αᵢ*) 
α,α*  - ½·ΣᵢΣⱼ(αᵢ - αᵢ*)(αⱼ - αⱼ*)K(xᵢ, xⱼ)

Subject to:
Σᵢ(αᵢ - αᵢ*) = 0
0 ≤ αᵢ, αᵢ* ≤ C
```

---

## Algorithm Complexity

- **Training Time**: O(n²·d) to O(n³·d)
  - n = samples (288)
  - d = features (4)
  - Dominated by kernel matrix computation

- **Prediction Time**: O(n_sv · d)
  - n_sv = number of support vectors (~120-150)
  - Very fast once trained

- **Space Complexity**: O(n_sv · d)
  - Only stores support vectors

---

## RBF Kernel Properties

1. **Universal Approximator**: Can approximate any continuous function
2. **Smooth Decision Boundary**: Creates smooth, non-linear boundaries
3. **Local Influence**: Points far apart have kernel value near 0
4. **Similarity Measure**: Higher values for similar points

**Kernel Value Range:**
```
0 < K(x, x') ≤ 1
K(x, x) = 1 (identical points)
K(x, x') → 0 as ||x - x'|| → ∞
```

---

## Epsilon-Tube Concept

The ε-insensitive loss function creates a "tube" around the predictions:

```
    |
    |     ε-tube
    |  ___________
    | /           \
    |/             \
    |\             /
    | \___________/
    |
    |________________
```

- **Inside tube** (|error| ≤ ε): No loss
- **Outside tube** (|error| > ε): Linear loss
- **Support vectors**: Points outside or on the tube boundary

---

## Support Vector Classification

**Three types of points:**

1. **Inside margin** (αᵢ = 0):
   - Not support vectors
   - Do not contribute to prediction
   - Well-predicted by the model

2. **On margin** (0 < αᵢ < C):
   - Margin support vectors
   - Define the epsilon-tube
   - Critical for model definition

3. **Outside margin** (αᵢ = C):
   - Error support vectors
   - Predictions outside ε-tube
   - May indicate outliers or difficult samples

---

## Conclusion

Support Vector Regression with RBF kernel achieved excellent performance (R² = 94.98%) on the slope stability prediction task. The model effectively uses ~40-50% of training samples as support vectors, providing a memory-efficient and accurate solution. Its robust handling of non-linear relationships makes it highly suitable for geotechnical applications.
