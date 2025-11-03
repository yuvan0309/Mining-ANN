# ALGORITHM USED FOR SUPPORT VECTOR MACHINE (SVM) REGRESSION

## Algorithm: Support Vector Regression (SVR) with RBF Kernel

### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

### Step 2: Initialize Hyperparameters
1. Set kernel type: RBF (Radial Basis Function)
   - RBF kernel: K(x, x') = exp(-γ||x - x'||²)
2. Initialize regularization parameter C = 1.0
   - Controls trade-off between model complexity and training error
3. Initialize kernel coefficient γ = 'scale'
   - γ = 1 / (n_features × variance(X))
4. Set epsilon ε = 0.1
   - Defines epsilon-tube for loss function

### Step 3: Feature Scaling
1. Create StandardScaler transformer
2. Fit scaler on training data:
   - μ (mean) = Σx / n
   - σ (std) = √(Σ(x - μ)² / n)
3. Transform features:
   - X_scaled = (X - μ) / σ
4. Apply same transformation to test data

### Step 4: Build SVM Model
1. Initialize SVR estimator with parameters:
   - kernel = 'rbf'
   - C = 1.0
   - gamma = 'scale'
   - epsilon = 0.1
2. Create pipeline combining scaler and SVR

### Step 5: Training Phase
For each training sample (xᵢ, yᵢ):

1. **Compute kernel matrix K:**
   - K[i,j] = exp(-γ||xᵢ - xⱼ||²)
   
2. **Solve optimization problem:**
   Minimize: ½||w||² + C·Σᵢ(ξᵢ + ξᵢ*)
   
   Subject to:
   - yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
   - (w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
   - ξᵢ, ξᵢ* ≥ 0
   
   Where:
   - w = weight vector
   - φ(x) = feature mapping
   - b = bias term
   - ξ = slack variables

3. **Find support vectors:**
   - Identify samples with 0 < αᵢ < C
   - These become support vectors

4. **Compute decision function:**
   f(x) = Σᵢ(αᵢ - αᵢ*)K(xᵢ, x) + b

### Step 6: Prediction Phase
For each new sample x:

1. Scale input features:
   - x_scaled = (x - μ) / σ

2. Compute kernel with all support vectors:
   - K(xⱼ, x) = exp(-γ||xⱼ - x||²)
   
3. Calculate prediction:
   - ŷ = Σⱼ(αⱼ - αⱼ*)K(xⱼ, x) + b
   
4. Return predicted FoS value

### Step 7: Model Evaluation
1. Make predictions on test set
2. Calculate R² Score:
   - R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
3. Calculate RMSE:
   - RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
4. Calculate MAE:
   - MAE = Σ|yᵢ - ŷᵢ| / n

### Step 8: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score

### Step 9: Save Model
1. Serialize trained pipeline using joblib
2. Save to 'svm.joblib'
3. Store performance metrics in JSON

### Step 10: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| kernel | 'rbf' | Radial Basis Function kernel |
| C | 1.0 | Regularization parameter |
| gamma | 'scale' | Kernel coefficient (auto-calculated) |
| epsilon | 0.1 | Epsilon-tube width |

## Performance Metrics

- **R² Score**: 0.9498 (94.98% variance explained)
- **RMSE**: 0.0451
- **MAE**: 0.0325
- **Status**: Best performing model

## Advantages
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Robust to outliers within epsilon-tube
- Non-linear modeling with RBF kernel

## Mathematical Formulation

**Decision Function:**
```
f(x) = Σᵢ₌₁ⁿ (αᵢ - αᵢ*) · K(xᵢ, x) + b
```

**RBF Kernel:**
```
K(x, x') = exp(-γ · ||x - x'||²)
where γ = 1 / (n_features · var(X))
```

**Loss Function (ε-insensitive):**
```
L(y, f(x)) = max(0, |y - f(x)| - ε)
```
