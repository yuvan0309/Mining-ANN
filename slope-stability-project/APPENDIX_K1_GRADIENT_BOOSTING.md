# APPENDIX K1

## ALGORITHM USED FOR GRADIENT BOOSTING REGRESSION

### Algorithm: Gradient Boosting Regressor with Ensemble Learning

---

## Step 1: Load and Prepare Data

1. Load the training dataset containing material properties (Cohesion, Friction Angle, Unit Weight, Ru) and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio, 288 training samples, 73 test samples)

---

## Step 2: Initialize Hyperparameters

1. **Set learning rate**: η = 0.1
   - Controls the contribution of each tree
   - Smaller values require more trees but improve generalization

2. **Number of estimators**: n_estimators = 100
   - Total number of boosting stages

3. **Maximum depth**: max_depth = 3
   - Maximum depth of individual regression trees
   - Controls model complexity

4. **Minimum samples split**: min_samples_split = 2
   - Minimum samples required to split an internal node

5. **Minimum samples leaf**: min_samples_leaf = 1
   - Minimum samples required at leaf node

6. **Loss function**: loss = 'squared_error'
   - L(y, F(x)) = (y - F(x))²/2

7. **Subsample**: subsample = 1.0
   - Fraction of samples used for fitting base learners

---

## Step 3: Feature Scaling

1. Create StandardScaler transformer
2. Fit scaler on training data:
   - μ (mean) = Σx / n
   - σ (std) = √(Σ(x - μ)² / n)
3. Transform features:
   - X_scaled = (X - μ) / σ
4. Apply same transformation to test data

**Feature Statistics:**
- Cohesion: μ ≈ 30.5 kPa, σ ≈ 15.2
- Friction Angle: μ ≈ 25.3°, σ ≈ 8.7
- Unit Weight: μ ≈ 20.1 kN/m³, σ ≈ 2.8
- Ru: μ ≈ 0.15, σ ≈ 0.12

---

## Step 4: Build Gradient Boosting Model

1. Initialize GradientBoostingRegressor with parameters:
   - learning_rate = 0.1
   - n_estimators = 100
   - max_depth = 3
   - min_samples_split = 2
   - min_samples_leaf = 1
   - loss = 'squared_error'
   - random_state = 42

2. Create pipeline combining scaler and regressor

---

## Step 5: Training Phase - Iterative Boosting

**Initialize:**
1. Set F₀(x) = ȳ (mean of training targets)
2. Initialize residuals: r₀ = y - F₀(x)

**For each boosting iteration m = 1 to M:**

1. **Compute pseudo-residuals:**
   ```
   rₘ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
   ```
   For squared error loss:
   ```
   rₘ = yᵢ - Fₘ₋₁(xᵢ)
   ```

2. **Fit regression tree hₘ(x) to residuals:**
   - Build CART (Classification and Regression Tree) with max_depth=3
   - Split nodes to minimize MSE:
     ```
     MSE = Σ(rᵢ - r̄)²/n
     ```
   - Find best split for each feature:
     ```
     Split = argmin[MSE_left + MSE_right]
     ```

3. **Compute leaf values γⱼₘ for each terminal node j:**
   ```
   γⱼₘ = argmin Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ)
   ```
   For squared error:
   ```
   γⱼₘ = mean(residuals in leaf j)
   ```

4. **Update model:**
   ```
   Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
   ```
   Where η = learning rate (0.1)

5. **Update residuals:**
   ```
   rₘ = yᵢ - Fₘ(xᵢ)
   ```

**Final Model:**
```
F(x) = F₀ + η · Σₘ₌₁ᴹ hₘ(x)
```

---

## Step 6: Tree Building Process (CART)

**For each tree at iteration m:**

1. **Node Splitting:**
   - For each feature j and split point s:
     ```
     Left: X[:, j] ≤ s
     Right: X[:, j] > s
     ```

2. **Impurity Calculation:**
   ```
   Impurity = (n_left/n) · MSE_left + (n_right/n) · MSE_right
   ```

3. **Best Split Selection:**
   ```
   (j*, s*) = argmin Impurity(j, s)
   ```

4. **Stopping Criteria:**
   - max_depth reached (depth = 3)
   - min_samples_split not satisfied (< 2)
   - min_samples_leaf not satisfied (< 1)
   - No further reduction in impurity

5. **Leaf Value Assignment:**
   ```
   Prediction at leaf = mean(target values in leaf)
   ```

---

## Step 7: Prediction Phase

For each new sample x:

1. **Scale input features:**
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Initialize prediction:**
   ```
   F(x) = F₀ = ȳ_train
   ```

3. **Accumulate predictions from all trees:**
   ```
   For m = 1 to 100:
       F(x) = F(x) + η · hₘ(x)
   ```

4. **Return final FoS prediction:**
   ```
   ŷ = F(x)
   ```

---

## Step 8: Model Evaluation

**Training Set Performance:**
1. Make predictions on training set
2. Calculate R² Score:
   ```
   R²_train = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   R²_train = 0.9954 (99.54%)
   ```

**Test Set Performance:**
1. Make predictions on test set
2. Calculate R² Score:
   ```
   R²_test = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   R²_test = 0.9426 (94.26%)
   ```

3. Calculate RMSE:
   ```
   RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
   RMSE = 0.0834
   ```

4. Calculate MAE:
   ```
   MAE = Σ|yᵢ - ŷᵢ| / n
   MAE = 0.0563
   ```

---

## Step 9: Cross-Validation

1. Split data into k=5 folds
2. For each fold i:
   - Train on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score:
   ```
   CV_mean = Σ(R²_fold) / k
   ```

---

## Step 10: Feature Importance Analysis

Calculate feature importance based on impurity reduction:

```
Importance(feature_j) = Σ (impurity_reduction from splits on feature_j)
```

**Feature Rankings:**
1. Friction Angle: ~40%
2. Cohesion: ~35%
3. Unit Weight: ~15%
4. Ru: ~10%

---

## Step 11: Save Model

1. Serialize trained pipeline using joblib
2. Save to 'best_model_gradient_boosting.pkl'
3. Save scaler to 'scaler.pkl'
4. Store performance metrics in 'results_summary.json'

---

## Step 12: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 0.1 | Step size shrinkage to prevent overfitting |
| n_estimators | 100 | Number of boosting stages |
| max_depth | 3 | Maximum depth of individual trees |
| min_samples_split | 2 | Minimum samples to split node |
| min_samples_leaf | 1 | Minimum samples at leaf |
| loss | 'squared_error' | Loss function to optimize |
| subsample | 1.0 | Fraction of samples for fitting |

---

## Performance Metrics

- **R² Score (Test)**: 0.9426 (94.26% variance explained)
- **R² Score (Training)**: 0.9954 (99.54% variance explained)
- **RMSE**: 0.0834
- **MAE**: 0.0563
- **Overfitting Gap**: 5.28%
- **Status**: **BEST PERFORMING MODEL** ⭐

---

## Advantages

1. **High Accuracy**: Best R² score among all models
2. **Sequential Learning**: Each tree corrects errors of previous trees
3. **Robust to Outliers**: Uses residual-based learning
4. **Feature Importance**: Provides interpretable feature rankings
5. **Non-linear Relationships**: Captures complex patterns
6. **Regularization**: Learning rate controls overfitting

---

## Disadvantages

1. **Training Time**: Sequential nature makes it slower
2. **Memory Usage**: Stores all trees in ensemble
3. **Hyperparameter Sensitivity**: Requires careful tuning
4. **Slight Overfitting**: 5.28% gap between train and test

---

## Mathematical Formulation

### Final Model:
```
F(x) = F₀ + Σₘ₌₁¹⁰⁰ η · hₘ(x)

Where:
- F₀ = initial prediction (mean of y)
- η = learning rate (0.1)
- hₘ(x) = mth regression tree
- M = 100 trees
```

### Loss Function (Squared Error):
```
L(y, F(x)) = ½(y - F(x))²
```

### Gradient (Pseudo-residuals):
```
-∂L/∂F = y - F(x)
```

### Impurity (MSE for regression):
```
MSE = (1/n)Σᵢ₌₁ⁿ(yᵢ - ȳ)²
```

### Tree Split Criterion:
```
Q(j, s) = (n_left/n)MSE_left + (n_right/n)MSE_right
Best split = argmin Q(j, s)
```

---

## Algorithm Complexity

- **Training Time**: O(n · m · d · f · log(n))
  - n = samples (288)
  - m = trees (100)
  - d = depth (3)
  - f = features (4)

- **Prediction Time**: O(m · d)
  - Very fast prediction

- **Space Complexity**: O(m · 2^d)
  - Stores all trees

---

## Conclusion

Gradient Boosting achieved the highest test accuracy (94.26%) with minimal overfitting, making it the **best model** for slope stability prediction. The ensemble of 100 shallow trees effectively captures non-linear relationships between geotechnical parameters and Factor of Safety.
