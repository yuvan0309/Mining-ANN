# ALGORITHM USED FOR XGBOOST (EXTREME GRADIENT BOOSTING)

## Algorithm: Gradient Boosting with Regularization

### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

### Step 2: Initialize Hyperparameters
1. Set number of boosting rounds: n_estimators = 100
2. Set learning rate: eta = 0.3 (shrinks feature weights)
3. Set maximum tree depth: max_depth = 6
4. Set L1 regularization: alpha = 0 (LASSO)
5. Set L2 regularization: lambda = 1 (Ridge)
6. Set subsample ratio: subsample = 1.0
7. Set feature sampling: colsample_bytree = 1.0
8. Set random state: random_state = 42

### Step 3: Initialize Model
1. Create initial prediction (base value):
   ```
   F₀(x) = ȳ = (1/n)·Σyᵢ
   ```
2. Set up residual storage for gradient computation

### Step 4: Training Phase - Boosting Iterations

For iteration m = 1 to n_estimators:

#### Step 4.1: Compute Pseudo-Residuals
1. Calculate gradient of loss function for each sample:
   ```
   rᵢₘ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ) = yᵢ - Fₘ₋₁(xᵢ)
   ```
   For squared error loss: residual = actual - predicted

2. Calculate Hessian (second derivative):
   ```
   hᵢₘ = ∂²L(yᵢ, F(xᵢ))/∂F(xᵢ)² = 1 (for squared error)
   ```

#### Step 4.2: Fit Regression Tree to Residuals

**Build decision tree using XGBoost's split finding:**

1. **For each node, find optimal split:**
   
   **Split Gain Calculation:**
   ```
   Gain = (GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ))/2 - γ
   
   Where:
   GL = Σ(gradients in left child)
   GR = Σ(gradients in right child)
   HL = Σ(hessians in left child)
   HR = Σ(hessians in right child)
   λ = L2 regularization parameter
   γ = complexity penalty (min_split_loss)
   ```

2. **Choose split that maximizes Gain:**
   - Try all features and thresholds
   - Select split with highest gain
   - If max_gain < 0, make it a leaf node

3. **Leaf Weight Calculation:**
   ```
   wⱼ = -Σ(gradients in leaf j) / (Σ(hessians in leaf j) + λ)
   ```

4. **Apply tree depth and sample constraints:**
   - Stop if depth ≥ max_depth
   - Stop if samples < min_child_weight

#### Step 4.3: Update Model
1. Add new tree to ensemble with learning rate:
   ```
   Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)
   
   Where:
   η = learning rate (0.3)
   hₘ(x) = prediction from tree m
   ```

2. Shrink contribution to reduce overfitting

#### Step 4.4: Apply Regularization
1. **L1 Regularization (α):**
   - Promotes sparsity in leaf weights
   - Can set some weights to exactly zero

2. **L2 Regularization (λ):**
   - Smooths leaf weights
   - Reduces magnitude of weights

3. **Total Objective Function:**
   ```
   Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₘ Ω(hₘ)
   
   Where regularization term:
   Ω(hₘ) = γT + (λ/2)Σⱼ wⱼ² + α Σⱼ |wⱼ|
   
   T = number of leaves
   wⱼ = weight of leaf j
   ```

### Step 5: Repeat Boosting
1. Repeat Step 4 for all n_estimators
2. Each tree corrects errors of previous trees
3. Model becomes more accurate with each iteration

### Step 6: Prediction Phase
For each new sample x:

1. **Initialize with base prediction:**
   ```
   F(x) = F₀(x) = ȳ
   ```

2. **Add contributions from all trees:**
   ```
   ŷ = F₀(x) + η·Σₘ₌₁ᴹ hₘ(x)
   ```
   Where M = n_estimators

3. Return final prediction ŷ

### Step 7: Feature Importance Calculation

**Three types of importance:**

1. **Weight:** Number of times feature appears in trees
2. **Gain:** Average gain when feature is used for splitting
   ```
   Importance_gain = Σ(Gain from splits using feature) / n_splits
   ```
3. **Cover:** Average coverage (samples affected) by feature

### Step 8: Model Evaluation
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

### Step 9: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train XGBoost on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score and standard deviation

### Step 10: Save Model
1. Serialize trained XGBoost model using joblib
2. Save to 'xgboost.joblib'
3. Store performance metrics and feature importances in JSON

### Step 11: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 100 | Number of boosting rounds |
| learning_rate (η) | 0.3 | Step size shrinkage |
| max_depth | 6 | Maximum tree depth |
| lambda (λ) | 1 | L2 regularization term |
| alpha (α) | 0 | L1 regularization term |
| gamma (γ) | 0 | Minimum loss reduction |
| subsample | 1.0 | Fraction of samples per tree |
| colsample_bytree | 1.0 | Fraction of features per tree |
| random_state | 42 | Random seed |

## Performance Metrics

- **R² Score**: 0.9204 (92.04% variance explained)
- **RMSE**: 0.0568
- **MAE**: 0.0422
- **Status**: Third best performing model

## Advantages
- Handles missing values automatically
- Built-in regularization (L1 + L2)
- Parallel tree construction (faster training)
- Handles non-linear relationships
- Provides multiple feature importance metrics
- Resistant to overfitting with proper regularization

## Mathematical Formulation

**Objective Function:**
```
Obj⁽ᵗ⁾ = Σᵢ₌₁ⁿ L(yᵢ, ŷᵢ⁽ᵗ⁾) + Σₖ₌₁ᵗ Ω(fₖ)

Loss Function:
L(y, ŷ) = (y - ŷ)²

Regularization:
Ω(f) = γT + (λ/2)Σⱼ₌₁ᵀ wⱼ² + α Σⱼ₌₁ᵀ |wⱼ|
```

**Taylor Expansion of Loss:**
```
L(yᵢ, ŷᵢ⁽ᵗ⁾) ≈ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + (1/2)hᵢfₜ²(xᵢ)

Where:
gᵢ = ∂L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾  (gradient)
hᵢ = ∂²L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾² (hessian)
```

**Optimal Leaf Weight:**
```
w*ⱼ = -Gⱼ / (Hⱼ + λ)

Where:
Gⱼ = Σᵢ∈Iⱼ gᵢ  (sum of gradients in leaf j)
Hⱼ = Σᵢ∈Iⱼ hᵢ  (sum of hessians in leaf j)
```

**Split Finding Gain:**
```
Gain = [GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)]/2 - γ
```

**Final Prediction:**
```
ŷᵢ = Σₖ₌₁ᴷ fₖ(xᵢ) = ŷᵢ⁽⁰⁾ + η·f₁(xᵢ) + η·f₂(xᵢ) + ... + η·fₖ(xᵢ)
```
