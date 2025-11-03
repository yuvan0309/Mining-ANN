# ALGORITHM USED FOR LIGHTGBM (LIGHT GRADIENT BOOSTING MACHINE)

## Algorithm: Gradient Boosting with Histogram-Based Learning

### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

### Step 2: Initialize Hyperparameters
1. Set number of boosting rounds: n_estimators = 100
2. Set learning rate: learning_rate = 0.1
3. Set maximum tree depth: max_depth = -1 (no limit, leaf-wise growth)
4. Set number of leaves: num_leaves = 31
5. Set L1 regularization: reg_alpha = 0
6. Set L2 regularization: reg_lambda = 0
7. Set minimum data in leaf: min_child_samples = 20
8. Set boosting type: boosting_type = 'gbdt' (Gradient Boosting Decision Tree)
9. Set random state: random_state = 42

### Step 3: Histogram Construction
1. **Bin continuous features into discrete bins:**
   - Default: max_bin = 255 bins per feature
   - For each feature, find bin boundaries using quantiles
   - Map feature values to bin indices

2. **Build feature histograms:**
   ```
   For each feature:
     - Sort values
     - Divide into max_bin intervals
     - Store bin boundaries
   ```

3. **Advantages of histogram approach:**
   - Reduced memory usage (uint8 instead of float32)
   - Faster split finding (O(bins) instead of O(data))
   - Better cache performance

### Step 4: Initialize Model
1. Create initial prediction (base value):
   ```
   F₀(x) = ȳ = (1/n)·Σyᵢ
   ```
2. Initialize gradient and hessian storage

### Step 5: Training Phase - Boosting Iterations

For iteration m = 1 to n_estimators:

#### Step 5.1: Compute Gradients and Hessians
1. Calculate first-order gradient (negative residual):
   ```
   gᵢ = ∂L(yᵢ, Fₘ₋₁(xᵢ))/∂Fₘ₋₁(xᵢ) = Fₘ₋₁(xᵢ) - yᵢ
   ```
   For squared error loss: gradient = predicted - actual

2. Calculate second-order gradient (hessian):
   ```
   hᵢ = ∂²L(yᵢ, Fₘ₋₁(xᵢ))/∂Fₘ₋₁(xᵢ)² = 1 (for squared error)
   ```

#### Step 5.2: Build Decision Tree (Leaf-wise Growth)

**LightGBM uses leaf-wise tree growth (Best-first) instead of level-wise:**

1. **Start with single leaf (root)**

2. **Iteratively split the leaf with maximum gain:**

**For each current leaf:**

**a) Histogram-based Split Finding:**

For each feature:
  - Use pre-built histogram bins
  - For each bin boundary as potential split:
    
    **Compute left and right statistics:**
    ```
    GL = Σ(gᵢ for samples with feature ≤ threshold)
    HL = Σ(hᵢ for samples with feature ≤ threshold)
    GR = Σ(gᵢ for samples with feature > threshold)
    HR = Σ(hᵢ for samples with feature > threshold)
    ```
    
    **Calculate split gain:**
    ```
    Gain = (GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ))/2 - γ
    
    Where:
    λ = reg_lambda (L2 regularization)
    γ = min_split_gain (complexity penalty)
    ```

**b) Select best split:**
  - Find split with maximum gain across all features
  - Store: (feature_index, threshold, gain)

**c) Split the best leaf:**
  - Among all current leaves, choose the one with maximum gain
  - Create two child leaves
  - Assign samples based on split condition

**d) Repeat until:**
  - Number of leaves = num_leaves (31), OR
  - Max_depth reached (if set), OR
  - No split has positive gain, OR
  - Min_child_samples constraint violated

#### Step 5.3: Gradient-based One-Side Sampling (GOSS)
Optional technique to speed up training:

1. **Sort samples by gradient magnitude |gᵢ|**
2. **Keep top a% samples** (large gradients)
3. **Randomly sample b% from remaining** (small gradients)
4. **Amplify small gradient samples** by factor (1-a)/b
5. Use this subset for tree building

#### Step 5.4: Exclusive Feature Bundling (EFB)
Optional technique to reduce features:

1. **Identify mutually exclusive features:**
   - Features that rarely take non-zero values simultaneously
   - E.g., one-hot encoded categorical variables

2. **Bundle features together:**
   - Combine into single feature
   - Add offsets to distinguish original features

3. **Reduces feature dimension** while maintaining accuracy

#### Step 5.5: Calculate Leaf Weights
For each leaf j:
```
wⱼ = -Gⱼ / (Hⱼ + λ)

Where:
Gⱼ = Σ(gradients of samples in leaf j)
Hⱼ = Σ(hessians of samples in leaf j)
λ = reg_lambda
```

#### Step 5.6: Update Model
Add new tree to ensemble:
```
Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)

Where:
η = learning_rate (0.1)
hₘ(x) = prediction from tree m
```

### Step 6: Regularization
1. **L1 Regularization (reg_alpha):**
   - Applied to leaf weights
   - Promotes sparsity

2. **L2 Regularization (reg_lambda):**
   - Applied to leaf weights
   - Smooths predictions

3. **Leaf-wise growth regularization:**
   - num_leaves controls model complexity
   - min_child_samples prevents overfitting

### Step 7: Repeat Boosting
1. Repeat Step 5 for all n_estimators
2. Each tree focuses on samples with large gradients
3. Model progressively reduces error

### Step 8: Prediction Phase
For each new sample x:

1. **Map feature values to histogram bins**

2. **Initialize with base prediction:**
   ```
   F(x) = F₀(x) = ȳ
   ```

3. **Traverse each tree:**
   - Start at root
   - Follow split conditions using binned features
   - Reach a leaf node
   - Get leaf weight wⱼ

4. **Aggregate predictions:**
   ```
   ŷ = F₀(x) + η·Σₘ₌₁ᴹ hₘ(x)
   ```

5. Return final prediction ŷ

### Step 9: Feature Importance Calculation

**Two types of importance:**

1. **Split-based importance:**
   - Count: number of times feature is used for splitting
   ```
   Importance_split = number of splits using feature
   ```

2. **Gain-based importance:**
   - Sum of gains when feature is used
   ```
   Importance_gain = Σ(gain from splits using feature)
   ```

### Step 10: Model Evaluation
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

### Step 11: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train LightGBM on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score and standard deviation

### Step 12: Save Model
1. Serialize trained LightGBM model using joblib
2. Save to 'lightgbm.joblib'
3. Store performance metrics and feature importances in JSON

### Step 13: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 100 | Number of boosting rounds |
| learning_rate | 0.1 | Shrinkage rate |
| max_depth | -1 | No depth limit (leaf-wise) |
| num_leaves | 31 | Maximum leaves per tree |
| reg_alpha | 0 | L1 regularization |
| reg_lambda | 0 | L2 regularization |
| min_child_samples | 20 | Min samples in leaf |
| boosting_type | 'gbdt' | Gradient Boosting Decision Tree |
| max_bin | 255 | Max bins per feature |
| random_state | 42 | Random seed |

## Performance Metrics

- **R² Score**: 0.9192 (91.92% variance explained)
- **RMSE**: 0.0572
- **MAE**: 0.0424
- **Status**: Fourth best performing model

## Advantages
- **Very fast training** (histogram-based + leaf-wise)
- **Memory efficient** (histogram binning)
- **Better accuracy** (leaf-wise growth finds optimal splits)
- **Handles large datasets** well
- **Built-in categorical feature** support
- **Gradient-based sampling** reduces computation
- **Feature bundling** reduces dimensionality

## Mathematical Formulation

**Objective Function:**
```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)

Loss Function:
L(y, ŷ) = (y - ŷ)²

Regularization:
Ω(f) = γT + (λ/2)Σⱼ wⱼ² + α Σⱼ |wⱼ|
```

**Gradient and Hessian:**
```
gᵢ = ∂L(yᵢ, ŷᵢ)/∂ŷᵢ = ŷᵢ - yᵢ
hᵢ = ∂²L(yᵢ, ŷᵢ)/∂ŷᵢ² = 1
```

**Optimal Leaf Weight:**
```
w*ⱼ = -Gⱼ / (Hⱼ + λ)

Where:
Gⱼ = Σᵢ∈Iⱼ gᵢ
Hⱼ = Σᵢ∈Iⱼ hᵢ
```

**Split Gain (Histogram-based):**
```
Gain = GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)

For bin boundary b:
GL = Σ(gᵢ for samples in bins ≤ b)
GR = Σ(gᵢ for samples in bins > b)
```

**Final Prediction:**
```
ŷ = F₀ + η·Σₘ₌₁ᴹ fₘ(x)

Where M = n_estimators, η = learning_rate
```

**Leaf-wise vs Level-wise Growth:**
```
Level-wise: Split all nodes at same depth
  → Balanced tree, slower convergence

Leaf-wise: Split leaf with maximum gain
  → Deeper tree, faster convergence, risk of overfitting
  → Controlled by num_leaves and max_depth
```
