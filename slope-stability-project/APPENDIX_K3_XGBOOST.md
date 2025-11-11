# APPENDIX K3

## ALGORITHM USED FOR XGBOOST REGRESSION

### Algorithm: Extreme Gradient Boosting (XGBoost) with Regularization

---

## Step 1: Load and Prepare Data

1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio, 288 training, 73 test)

---

## Step 2: Initialize Hyperparameters

1. **Objective function**: objective = 'reg:squarederror'
   - Regression with squared error loss

2. **Learning rate (eta)**: η = 0.1
   - Step size shrinkage for updates

3. **Number of estimators**: n_estimators = 100
   - Number of boosting rounds

4. **Maximum depth**: max_depth = 3
   - Maximum tree depth for base learners

5. **Minimum child weight**: min_child_weight = 1
   - Minimum sum of instance weight in child

6. **Gamma (min split loss)**: γ = 0
   - Minimum loss reduction for split

7. **Subsample**: subsample = 0.8
   - Fraction of samples for tree training

8. **Column sample by tree**: colsample_bytree = 0.8
   - Fraction of features for each tree

9. **L1 regularization (alpha)**: α = 0
   - L1 weight regularization term

10. **L2 regularization (lambda)**: λ = 1
    - L2 weight regularization term

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

## Step 4: Build XGBoost Model

1. Initialize XGBRegressor with parameters
2. Create pipeline combining scaler and XGBoost
3. Set tree method: 'auto' (GPU if available, else hist)
4. Enable parallel processing

---

## Step 5: Training Phase - Regularized Boosting

**Initialize:**
```
F₀(x) = argmin Σᵢ L(yᵢ, c)
For squared error: F₀(x) = ȳ
```

**For each boosting round t = 1 to T:**

1. **Compute first-order gradient (gᵢ):**
   ```
   gᵢ = ∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
   
   For squared error:
   gᵢ = F(xᵢ) - yᵢ
   ```

2. **Compute second-order gradient (hᵢ):**
   ```
   hᵢ = ∂²L(yᵢ, F(xᵢ))/∂F²(xᵢ)
   
   For squared error:
   hᵢ = 1
   ```

3. **Build tree to minimize objective:**
   ```
   Obj⁽ᵗ⁾ = Σᵢ L(yᵢ, Fₜ₋₁(xᵢ) + fₜ(xᵢ)) + Ω(fₜ)
   
   Where Ω(f) = γT + ½λΣⱼwⱼ² + α·Σⱼ|wⱼ|
   
   T = number of leaves
   wⱼ = leaf weights
   ```

4. **Approximate objective using Taylor expansion:**
   ```
   Obj⁽ᵗ⁾ ≈ Σᵢ[gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)
   ```

5. **For each leaf j, calculate optimal weight:**
   ```
   wⱼ* = -Σᵢ∈Iⱼ gᵢ / (Σᵢ∈Iⱼ hᵢ + λ)
   
   Where Iⱼ = instances in leaf j
   ```

6. **Calculate gain for splitting node:**
   ```
   Gain = ½[(Σᵢ∈Iₗ gᵢ)² / (Σᵢ∈Iₗ hᵢ + λ) 
           + (Σᵢ∈Iᵣ gᵢ)² / (Σᵢ∈Iᵣ hᵢ + λ)
           - (Σᵢ∈I gᵢ)² / (Σᵢ∈I hᵢ + λ)] - γ
   ```

7. **Find best split:**
   ```
   For each feature j and split point s:
       Calculate Gain(j, s)
       Track maximum gain split
   ```

8. **Apply split if Gain > 0**, otherwise create leaf

9. **Update model:**
   ```
   Fₜ(x) = Fₜ₋₁(x) + η · fₜ(x)
   ```

**Final Model:**
```
F(x) = F₀ + Σₜ₌₁ᵀ η · fₜ(x)
```

---

## Step 6: Tree Growing Algorithm

**XGBoost Level-wise Growth:**

1. **Start with root node** containing all samples

2. **For each level up to max_depth:**
   
   a. **For each leaf node at current level:**
      - Evaluate all possible splits
      - Calculate gain for each split
      - Select best split with maximum gain
   
   b. **Split selection criteria:**
      ```
      - Gain > 0
      - Depth < max_depth
      - Σhᵢ > min_child_weight for both children
      ```
   
   c. **Create child nodes** if criteria met

3. **Stop when:**
   - max_depth reached
   - No positive gain splits
   - min_child_weight constraint violated

---

## Step 7: Regularization Components

1. **L2 Regularization (Ridge):**
   ```
   λ · Σⱼwⱼ²
   Penalizes large leaf weights
   ```

2. **L1 Regularization (Lasso):**
   ```
   α · Σⱼ|wⱼ|
   Promotes sparsity in leaf weights
   ```

3. **Tree Complexity:**
   ```
   γ · T
   Penalizes number of leaves
   ```

4. **Min Child Weight:**
   ```
   Prevents creating child nodes with insufficient data
   ```

---

## Step 8: Prediction Phase

**For each new sample x:**

1. **Scale input features:**
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Initialize prediction:**
   ```
   F(x) = F₀ = ȳ_train
   ```

3. **Traverse each tree:**
   ```
   For t = 1 to 100:
       Start at root of tree t
       While not leaf:
           If x[feature] ≤ threshold:
               Go to left child
           Else:
               Go to right child
       Add η · leaf_value to F(x)
   ```

4. **Return final prediction:**
   ```
   ŷ = F(x)
   ```

---

## Step 9: Model Evaluation

**Test Set Performance:**
1. Calculate R² Score:
   ```
   R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   R² = 0.9420 (94.20%)
   ```

2. Calculate RMSE:
   ```
   RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
   RMSE = 0.0838
   ```

3. Calculate MAE:
   ```
   MAE = Σ|yᵢ - ŷᵢ| / n
   MAE = 0.0597
   ```

**Training Set Performance:**
```
R²_train = 0.9581 (95.81%)
```

**Generalization:**
```
Overfitting Gap = R²_train - R²_test = 1.61%
Status: Excellent generalization
```

---

## Step 10: Feature Importance

XGBoost provides multiple importance metrics:

1. **Gain**: Average gain of splits using the feature
2. **Cover**: Average coverage of splits using the feature
3. **Weight**: Number of times feature appears in trees

**Feature Rankings (by gain):**
1. Friction Angle: ~42%
2. Cohesion: ~36%
3. Unit Weight: ~14%
4. Ru: ~8%

---

## Step 11: Cross-Validation

1. Perform 5-fold cross-validation
2. Calculate mean CV score
3. Monitor validation performance during training
4. Detect overfitting early

---

## Step 12: Save Model

1. Serialize trained model using joblib
2. Save to 'best_model_xgboost.pkl'
3. Save scaler separately
4. Store feature importance
5. Record performance metrics

---

## Step 13: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| objective | 'reg:squarederror' | Loss function |
| learning_rate | 0.1 | Step size shrinkage |
| n_estimators | 100 | Number of trees |
| max_depth | 3 | Maximum tree depth |
| min_child_weight | 1 | Minimum sum of weights in child |
| gamma | 0 | Minimum loss reduction for split |
| subsample | 0.8 | Row sampling ratio |
| colsample_bytree | 0.8 | Column sampling ratio per tree |
| reg_alpha | 0 | L1 regularization |
| reg_lambda | 1 | L2 regularization |

---

## Performance Metrics

- **R² Score (Test)**: 0.9420 (94.20% variance explained)
- **R² Score (Training)**: 0.9581 (95.81% variance explained)
- **RMSE**: 0.0838
- **MAE**: 0.0597
- **Overfitting Gap**: 1.61% (Minimal - **Best generalization**)
- **Status**: Second best overall, **Best generalization**
- **Training Time**: Fast (optimized C++ implementation)

---

## Advantages

1. **Excellent Generalization**: Only 1.61% overfitting gap
2. **Regularization**: Built-in L1, L2, and tree complexity penalties
3. **Speed**: Highly optimized, parallel processing
4. **Sparsity Aware**: Handles missing values automatically
5. **Feature Importance**: Multiple importance metrics
6. **Flexibility**: Many tuning options
7. **Robustness**: Less prone to overfitting than Gradient Boosting

---

## Disadvantages

1. **Many Hyperparameters**: Requires careful tuning
2. **Black Box**: Less interpretable than single trees
3. **Memory**: Requires more memory than simple models
4. **Slight Lower Accuracy**: 0.06% below Gradient Boosting

---

## Mathematical Formulation

### Objective Function:
```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

Where:
L = Loss function (squared error)
Ω = Regularization term
```

### Regularization Term:
```
Ω(f) = γT + ½λΣⱼwⱼ² + α·Σⱼ|wⱼ|

Where:
γ = complexity parameter
T = number of leaves
λ = L2 regularization
α = L1 regularization
wⱼ = leaf weights
```

### Taylor Expansion Approximation:
```
L(yᵢ, Fₜ₋₁ + fₜ) ≈ L(yᵢ, Fₜ₋₁) + gᵢfₜ + ½hᵢfₜ²

Where:
gᵢ = ∂L/∂F (first-order gradient)
hᵢ = ∂²L/∂F² (second-order gradient)
```

### Optimal Leaf Weight:
```
wⱼ* = -Gⱼ / (Hⱼ + λ)

Where:
Gⱼ = Σᵢ∈Iⱼ gᵢ (sum of gradients in leaf)
Hⱼ = Σᵢ∈Iⱼ hᵢ (sum of hessians in leaf)
```

### Split Gain:
```
Gain = ½[Gₗ²/(Hₗ + λ) + Gᵣ²/(Hᵣ + λ) - G²/(H + λ)] - γ

Where:
Gₗ, Hₗ = left child gradient/hessian sums
Gᵣ, Hᵣ = right child gradient/hessian sums
G, H = parent node gradient/hessian sums
```

---

## Algorithm Complexity

- **Training Time**: O(n · d · K · log(n))
  - n = samples (288)
  - d = features (4)
  - K = trees (100)
  - Highly optimized with parallelization

- **Prediction Time**: O(K · log(T))
  - K = trees (100)
  - T = average leaves per tree (~8)
  - Very fast prediction

- **Space Complexity**: O(K · T · d)
  - Efficient tree representation

---

## XGBoost vs Gradient Boosting

| Aspect | XGBoost | Gradient Boosting |
|--------|---------|-------------------|
| Regularization | L1 + L2 + Complexity | Learning rate only |
| Speed | Very fast (parallel) | Slower (sequential) |
| Overfitting | 1.61% gap | 5.28% gap |
| Test R² | 94.20% | 94.26% |
| Implementation | Optimized C++ | Python/scikit-learn |
| Missing values | Native handling | Requires imputation |

---

## Second-Order Optimization

XGBoost uses both first and second derivatives:

**Advantages:**
1. **Faster convergence**: Uses curvature information
2. **Better approximation**: Taylor expansion to 2nd order
3. **Adaptive learning**: Hessian-based weight scaling
4. **Numerical stability**: Better handling of edge cases

**Mathematical Intuition:**
```
Newton's method update:
w_new = w_old - [∂²L/∂w²]⁻¹ · [∂L/∂w]

Translated to XGBoost leaf weights:
wⱼ = -Σgᵢ / Σhᵢ
```

---

## Conclusion

XGBoost achieved outstanding performance (R² = 94.20%) with the **best generalization** among all models (only 1.61% overfitting gap). Its advanced regularization techniques and second-order optimization make it highly reliable for production deployment in slope stability prediction. The model's speed, accuracy, and robustness make it an excellent choice alongside Gradient Boosting.
