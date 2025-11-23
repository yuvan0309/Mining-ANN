# MACHINE LEARNING ALGORITHMS FOR SLOPE STABILITY PREDICTION

This document contains detailed algorithms for all models used in the slope stability prediction project.

---

## TABLE OF CONTENTS

1. [Gradient Boosting Regression](#1-gradient-boosting-regression)
2. [XGBoost (Extreme Gradient Boosting)](#2-xgboost-extreme-gradient-boosting)
3. [Random Forest Regression](#3-random-forest-regression)
4. [LightGBM (Light Gradient Boosting Machine)](#4-lightgbm-light-gradient-boosting-machine)
5. [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
6. [Artificial Neural Network (ANN)](#6-artificial-neural-network-ann)

---

## 1. GRADIENT BOOSTING REGRESSION

**Performance**: R² = 0.9426 (Test), 0.9954 (Train) | **Rank**: 🥇 1st

### ALGORITHM USED FOR GRADIENT BOOSTING REGRESSION

#### Step 1: Load and Prepare Data
1. Load the training dataset containing:
   - Features (X): Cohesion (c), Friction angle (φ), Unit weight (γ), Pore pressure ratio (Ru)
   - Target (y): Factor of Safety (FoS)
2. Split data into training (80%) and test sets (20%)

#### Step 2: Initialize Hyperparameters
```
learning_rate = 0.1
n_estimators = 100
max_depth = 3
min_samples_split = 2
min_samples_leaf = 1
loss = 'squared_error'
random_state = 42
```

#### Step 3: Feature Scaling
1. Create StandardScaler instance
2. Fit scaler on training data:
   ```
   μ = mean(X_train)
   σ = std(X_train)
   ```
3. Transform training and test data:
   ```
   X_scaled = (X - μ) / σ
   ```

#### Step 4: Initialize Base Prediction
1. Calculate initial prediction (mean of target):
   ```
   F₀(x) = ȳ = (1/n) · Σyᵢ
   ```
2. Initialize residuals:
   ```
   rᵢ = yᵢ - F₀(x)
   ```

#### Step 5: Build Sequential Trees
**For each tree m = 1 to n_estimators:**

1. **Compute negative gradient (residuals):**
   ```
   rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)
   ```
   Where Fₘ₋₁ is prediction from previous iteration

2. **Fit decision tree hₘ(x) to residuals:**
   - Input: (X, rᵢₘ)
   - Constraints:
     - max_depth = 3 (tree depth)
     - min_samples_split = 2
     - min_samples_leaf = 1

3. **For each terminal node j in tree m:**
   Calculate optimal leaf value:
   ```
   γⱼₘ = argmin Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ)
   
   For squared error:
   γⱼₘ = mean(residuals in leaf j)
   ```

4. **Update model:**
   ```
   Fₘ(x) = Fₘ₋₁(x) + learning_rate · hₘ(x)
   ```

#### Step 6: Prediction Phase
For new sample x:
```
ŷ = F₀(x) + learning_rate · Σₘ₌₁ᴹ hₘ(x)
```
Where M = n_estimators = 100

#### Step 7: Model Evaluation

1. **Calculate R² Score:**
   ```
   SS_res = Σ(yᵢ - ŷᵢ)²
   SS_tot = Σ(yᵢ - ȳ)²
   R² = 1 - (SS_res / SS_tot)
   ```

2. **Calculate RMSE:**
   ```
   RMSE = √[(1/n) · Σ(yᵢ - ŷᵢ)²]
   ```

3. **Calculate MAE:**
   ```
   MAE = (1/n) · Σ|yᵢ - ŷᵢ|
   ```

#### Step 8: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score and standard deviation

#### Step 9: Save Model
1. Save trained model: `best_model_gradient_boosting.pkl`
2. Save scaler: `scaler.pkl`
3. Save metrics for comparison

---

## 2. XGBOOST (EXTREME GRADIENT BOOSTING)

**Performance**: R² = 0.9420 (Test), 0.9581 (Train) | **Rank**: 🥈 2nd

### ALGORITHM USED FOR XGBOOST REGRESSION

#### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training (80%) and test sets (20%)

#### Step 2: Initialize Fine-Tuned Hyperparameters
```python
n_estimators = 300      # Increased from 200 for better learning
max_depth = 6           # Reduced from 10 to prevent overfitting
learning_rate = 0.05    # Reduced from 0.1 for better generalization
subsample = 0.8         # Use 80% of samples per tree
colsample_bytree = 0.8  # Use 80% of features per tree
min_child_weight = 3    # Increased from 1 to prevent overfitting
gamma = 0.1             # Minimum loss reduction for split
reg_alpha = 0.1         # L1 regularization (LASSO)
reg_lambda = 1.0        # L2 regularization (Ridge)
random_state = 42
```

#### Step 3: Feature Scaling
1. Create StandardScaler instance
2. Fit scaler on training data and transform both sets

#### Step 4: Initialize Model
1. Create initial prediction (base value):
   ```
   F₀(x) = ȳ = (1/n) · Σyᵢ
   ```
2. Set up regularized objective function:
   ```
   Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₘ)
   
   Where:
   Ω(f) = γT + (λ/2)·Σwⱼ² + α·Σ|wⱼ|
   
   T = number of leaves
   wⱼ = leaf weights
   ```

#### Step 5: Build Boosted Trees (Regularized)
**For each tree m = 1 to n_estimators:**

1. **Compute first and second order gradients:**
   ```
   gᵢ = ∂L(yᵢ, ŷᵢ⁽ᵐ⁻¹⁾) / ∂ŷ
   hᵢ = ∂²L(yᵢ, ŷᵢ⁽ᵐ⁻¹⁾) / ∂ŷ²
   
   For squared error:
   gᵢ = ŷᵢ⁽ᵐ⁻¹⁾ - yᵢ
   hᵢ = 1
   ```

2. **Find best split for each node:**
   - Sample subsample fraction of data
   - Sample colsample_bytree fraction of features
   - For each feature:
     ```
     Gain = (Σgᵢ)² / (Σhᵢ + λ) - γ
     ```
   - Choose split with maximum gain > gamma

3. **Calculate optimal leaf weights:**
   ```
   wⱼ* = -(Σgᵢ) / (Σhᵢ + λ)
   ```

4. **Update predictions:**
   ```
   ŷᵢ⁽ᵐ⁾ = ŷᵢ⁽ᵐ⁻¹⁾ + η · wⱼ*
   ```
   Where η = learning_rate = 0.05

#### Step 6: Prediction Phase
For each new sample x:
```
ŷ = F₀(x) + η · Σₘ₌₁ᴹ hₘ(x)
```
Where M = n_estimators = 300

#### Step 7: Regularization Benefits
1. **Tree Complexity Penalty (γ)**: Prevents creating too many leaves
2. **L2 Regularization (λ)**: Smooths leaf weights
3. **L1 Regularization (α)**: Promotes sparsity
4. **Subsampling**: Prevents overfitting by using subset of data
5. **Column Sampling**: Reduces feature correlation

#### Step 8: Model Evaluation
Calculate R², RMSE, MAE on test set

#### Step 9: Cross-Validation
5-fold cross-validation to assess generalization

#### Step 10: Save Model
Save as `best_model_xgboost.pkl`

---

## 3. RANDOM FOREST REGRESSION

**Performance**: R² = 0.9924 (Train) | **Rank**: 🥈 2nd (Training)

### ALGORITHM USED FOR RANDOM FOREST REGRESSION

#### Step 1: Load and Prepare Data
1. Load the training and test data
2. Separate features (X) from target (y)

#### Step 2: Initialize Hyperparameters
```
n_estimators = 200    # Number of trees in forest
max_depth = 15        # Maximum depth of each tree
min_samples_split = 2 # Minimum samples to split node
min_samples_leaf = 1  # Minimum samples in leaf
random_state = 42
```

#### Step 3: Feature Scaling
Apply StandardScaler to normalize features

#### Step 4: Build Random Forest
**For each tree t = 1 to n_estimators:**

1. **Create Bootstrap Sample:**
   ```
   Sample n observations with replacement
   Bootstrap_t = random_sample(X_train, n, replace=True)
   ```

2. **Start with all bootstrap samples at root node**

3. **For each node to split:**
   
   a. **Randomly select m features:**
      ```
      m = √(number_of_features)  # For regression
      features_subset = random_sample(features, m)
      ```
   
   b. **Find best split using RMSE criterion:**
      ```
      For each feature f in features_subset:
        For each threshold t:
          Split data: left = X[f ≤ t], right = X[f > t]
          
          Calculate weighted RMSE:
          RMSE = (n_left/n)·RMSE_left + (n_right/n)·RMSE_right
          
      Choose split with minimum RMSE
      ```
   
   c. **Split the data into child nodes**

4. **Stop splitting when:**
   - Maximum depth reached (max_depth = 15)
   - Node has < min_samples_split samples
   - All samples have same target value
   - Pure leaf node created

5. **Assign leaf node predictions:**
   ```
   prediction = mean(y_samples_in_leaf)
   ```

#### Step 5: Prediction Phase
For new sample x:

1. **Pass through all trees:**
   ```
   predictions = [tree_t.predict(x) for t in 1 to n_estimators]
   ```

2. **Compute average prediction:**
   ```
   ŷ = (1/n_estimators) · Σpredictions
   ```

#### Step 6: Feature Importance
Calculate importance based on reduction in RMSE:
```
For each feature f:
  importance_f = Σ (RMSE_before_split - RMSE_after_split)
  
Normalize so Σ importance = 1
```

#### Step 7: Model Evaluation
1. Calculate R² Score
2. Calculate RMSE
3. Calculate MAE
4. Perform 5-fold cross-validation

#### Step 8: End
Model trained and ready for deployment

---

## 4. LIGHTGBM (LIGHT GRADIENT BOOSTING MACHINE)

**Performance**: R² = 0.9872 (Train) | **Rank**: 🥉 3rd

### ALGORITHM USED FOR LIGHTGBM REGRESSION

#### Step 1: Load and Prepare Data
1. Load training and test datasets
2. Separate features (X) from target (y)
3. Split into 80% training, 20% test

#### Step 2: Initialize Hyperparameters
```python
num_leaves = 31           # Maximum tree leaves
learning_rate = 0.05      # Shrinkage rate
n_estimators = 100        # Number of boosting rounds
max_depth = -1            # No limit (-1)
min_child_samples = 20    # Minimum samples in leaf
subsample = 0.8           # Row sampling ratio
colsample_bytree = 0.8    # Column sampling ratio
random_state = 42
```

#### Step 3: Feature Scaling
Apply StandardScaler to features

#### Step 4: Initialize Base Prediction
```
F₀(x) = ȳ = mean(y_train)
```

#### Step 5: Build Gradient Boosting Trees (Leaf-wise)
**For each iteration m = 1 to n_estimators:**

1. **Compute gradients:**
   ```
   gᵢ = ∂L(yᵢ, ŷᵢ) / ∂ŷ = ŷᵢ - yᵢ
   hᵢ = ∂²L(yᵢ, ŷᵢ) / ∂ŷ² = 1
   ```

2. **Build histogram-based tree:**
   
   a. **Bin continuous features into histograms**
   
   b. **Find best split using Gradient-based One-Side Sampling (GOSS):**
      - Keep all large gradient samples
      - Randomly sample small gradient samples
      - Calculate gain for each split candidate
   
   c. **Split criterion (best leaf-wise split):**
      ```
      Gain = (Σgᵢ_left)² / (Σhᵢ_left) + (Σgᵢ_right)² / (Σhᵢ_right) - (Σgᵢ)² / (Σhᵢ)
      ```
   
   d. **Grow tree leaf-wise (not level-wise):**
      - Always split the leaf with maximum gain
      - Stop when num_leaves reached or min_child_samples violated

3. **Calculate optimal leaf weights:**
   ```
   wⱼ = -(Σgᵢ) / (Σhᵢ + λ)
   ```

4. **Update predictions:**
   ```
   Fₘ(x) = Fₘ₋₁(x) + learning_rate · hₘ(x)
   ```

#### Step 6: Exclusive Feature Bundling (EFB)
LightGBM bundles mutually exclusive features to reduce dimensions

#### Step 7: Prediction Phase
For new sample x:
```
ŷ = F₀(x) + learning_rate · Σₘ₌₁ᴹ hₘ(x)
```

#### Step 8: Model Evaluation
Calculate R², RMSE, MAE on test set

#### Step 9: Cross-Validation
Perform 5-fold cross-validation

#### Step 10: End
Model ready for predictions

---

## 5. SUPPORT VECTOR MACHINE (SVM)

**Performance**: R² = 0.9570 (Train) | **Rank**: 5th

### ALGORITHM USED FOR SVM REGRESSION (SVR)

#### Step 1: Load and Prepare Data
1. Load training and test datasets
2. Separate features (X) from target (y)

#### Step 2: Initialize Hyperparameters
```python
kernel = 'rbf'      # Radial Basis Function
C = 100             # Regularization parameter
gamma = 'scale'     # Kernel coefficient (1/(n_features * X.var()))
epsilon = 0.1       # Epsilon-tube width
```

#### Step 3: Feature Scaling
**Critical for SVM - must scale features:**
```
X_scaled = (X - μ) / σ
```

#### Step 4: Define RBF Kernel Function
```
K(x, x') = exp(-γ · ||x - x'||²)

Where:
γ = 1 / (n_features · var(X))  # For gamma='scale'
```

#### Step 5: Solve ε-SVR Optimization Problem

**Primal Form:**
```
minimize: (1/2)||w||² + C · Σ(ξᵢ + ξᵢ*)

subject to:
  yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
  (w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
  ξᵢ, ξᵢ* ≥ 0
```

Where:
- w = weight vector in feature space
- φ(x) = feature mapping (implicit via kernel)
- b = bias term
- ε = epsilon (insensitivity tube)
- ξᵢ, ξᵢ* = slack variables
- C = penalty for violations

**Dual Form (solved in practice):**
```
maximize: Σyᵢ(αᵢ - αᵢ*) - ε·Σ(αᵢ + αᵢ*) - (1/2)·ΣΣ(αᵢ - αᵢ*)(αⱼ - αⱼ*)K(xᵢ, xⱼ)

subject to:
  0 ≤ αᵢ, αᵢ* ≤ C
  Σ(αᵢ - αᵢ*) = 0
```

#### Step 6: Identify Support Vectors
Support vectors are samples where:
```
αᵢ > 0 or αᵢ* > 0
```

These are samples on or outside the ε-tube

#### Step 7: Prediction Phase
For new sample x:
```
ŷ = Σ(αᵢ - αᵢ*) · K(xᵢ, x) + b

Where sum is only over support vectors
```

#### Step 8: Model Evaluation
1. Calculate R² Score
2. Calculate RMSE
3. Calculate MAE
4. Perform cross-validation

#### Step 9: End
SVM model trained and ready

---

## 6. ARTIFICIAL NEURAL NETWORK (ANN)

**Performance**: R² = 0.9316 (Train) | **Rank**: 6th

### ALGORITHM USED FOR ANN REGRESSION (MLP)

#### Step 1: Load and Prepare Data
1. Load training and test datasets
2. Separate features (X) from target (y)

#### Step 2: Initialize Network Architecture
```python
hidden_layer_sizes = (64, 32, 16)  # 3 hidden layers
activation = 'relu'                 # ReLU activation
solver = 'adam'                     # Adam optimizer
learning_rate_init = 0.001         # Initial learning rate
max_iter = 1000                    # Maximum epochs
random_state = 42
```

**Network Structure:**
```
Input Layer:    4 neurons (c, φ, γ, Ru)
Hidden Layer 1: 64 neurons + ReLU
Hidden Layer 2: 32 neurons + ReLU
Hidden Layer 3: 16 neurons + ReLU
Output Layer:   1 neuron (FoS)
```

#### Step 3: Feature Scaling
```
X_scaled = (X - μ) / σ
```

#### Step 4: Initialize Weights and Biases
For each layer l:
```
W⁽ˡ⁾ ~ N(0, √(2/nᵢₙ))  # He initialization
b⁽ˡ⁾ = 0
```

#### Step 5: Forward Propagation

**For each training sample x:**

1. **Input to Hidden Layer 1:**
   ```
   z⁽¹⁾ = W⁽¹⁾·x + b⁽¹⁾
   a⁽¹⁾ = ReLU(z⁽¹⁾) = max(0, z⁽¹⁾)
   ```

2. **Hidden Layer 1 to Hidden Layer 2:**
   ```
   z⁽²⁾ = W⁽²⁾·a⁽¹⁾ + b⁽²⁾
   a⁽²⁾ = ReLU(z⁽²⁾)
   ```

3. **Hidden Layer 2 to Hidden Layer 3:**
   ```
   z⁽³⁾ = W⁽³⁾·a⁽²⁾ + b⁽³⁾
   a⁽³⁾ = ReLU(z⁽³⁾)
   ```

4. **Hidden Layer 3 to Output:**
   ```
   z⁽⁴⁾ = W⁽⁴⁾·a⁽³⁾ + b⁽⁴⁾
   ŷ = z⁽⁴⁾  # Linear activation for regression
   ```

#### Step 6: Compute Loss
```
L = (1/2n) · Σ(yᵢ - ŷᵢ)²  # Mean Squared Error
```

#### Step 7: Backward Propagation (Gradient Computation)

1. **Output layer gradient:**
   ```
   δ⁽⁴⁾ = ŷ - y
   ```

2. **Hidden layer 3 gradient:**
   ```
   δ⁽³⁾ = (W⁽⁴⁾)ᵀ·δ⁽⁴⁾ ⊙ ReLU'(z⁽³⁾)
   
   Where ReLU'(z) = 1 if z > 0, else 0
   ```

3. **Hidden layer 2 gradient:**
   ```
   δ⁽²⁾ = (W⁽³⁾)ᵀ·δ⁽³⁾ ⊙ ReLU'(z⁽²⁾)
   ```

4. **Hidden layer 1 gradient:**
   ```
   δ⁽¹⁾ = (W⁽²⁾)ᵀ·δ⁽²⁾ ⊙ ReLU'(z⁽¹⁾)
   ```

5. **Compute weight and bias gradients:**
   ```
   ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
   ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
   ```

#### Step 8: Update Weights using Adam Optimizer

For each parameter θ (weights and biases):

1. **Compute momentum (first moment):**
   ```
   m_t = β₁·m_{t-1} + (1-β₁)·g_t
   
   Where:
   g_t = ∂L/∂θ
   β₁ = 0.9
   ```

2. **Compute velocity (second moment):**
   ```
   v_t = β₂·v_{t-1} + (1-β₂)·g_t²
   
   Where β₂ = 0.999
   ```

3. **Bias correction:**
   ```
   m̂_t = m_t / (1 - β₁ᵗ)
   v̂_t = v_t / (1 - β₂ᵗ)
   ```

4. **Update parameters:**
   ```
   θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
   
   Where:
   α = learning_rate = 0.001
   ε = 10⁻⁸
   ```

#### Step 9: Iterate
Repeat Steps 5-8 for max_iter epochs or until convergence

#### Step 10: Prediction Phase
For new sample x:
1. Apply forward propagation (Step 5)
2. Return output ŷ

#### Step 11: Model Evaluation
Calculate R², RMSE, MAE on test set

#### Step 12: End
Neural network trained and ready

---

## MODEL COMPARISON SUMMARY

| Model | Algorithm Type | Test R² | Train R² | RMSE | MAE | Overfitting Gap |
|-------|---------------|---------|----------|------|-----|-----------------|
| **Gradient Boosting** | Sequential Ensemble | **94.26%** | 99.54% | 0.0834 | **0.0563** | 5.28% |
| **XGBoost** | Regularized Boosting | 94.20% | 95.81% | 0.0838 | 0.0597 | **1.61%** ✓ |
| **Random Forest** | Parallel Ensemble | - | 99.24% | 0.0313 | 0.0220 | Not tested |
| **LightGBM** | Leaf-wise Boosting | - | 98.72% | 0.0407 | 0.0297 | Not tested |
| **SVM** | Kernel Method | - | 95.70% | 0.0746 | 0.0616 | Not tested |
| **ANN** | Neural Network | - | 93.16% | 0.0940 | 0.0694 | Not tested |

---

## KEY INSIGHTS

### Why Gradient Boosting Won:
✅ Highest test accuracy (94.26%)  
✅ Lowest MAE (0.0563)  
✅ Best overall performance  

### Why XGBoost is Close Second:
✅ Best generalization (1.61% gap)  
✅ Regularization prevents overfitting  
✅ Nearly identical test R² (94.20%)  
✅ Production-ready with robust predictions  

### Production Deployment:
Both Gradient Boosting and XGBoost are deployed in the web application for ensemble predictions.

---

**Generated**: November 2025  
**Project**: Slope Stability Prediction using Machine Learning  
**Method**: Bishop's Simplified Method  
**Dataset**: 361 samples (80% train, 20% test)  
**Validation**: 5-fold cross-validation  

---
