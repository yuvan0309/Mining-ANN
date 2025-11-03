# ALGORITHM USED FOR RANDOM FOREST REGRESSION

## Algorithm: Random Forest Ensemble Learning

### Step 1: Load and Prepare Data
1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio)

### Step 2: Initialize Hyperparameters
1. Set number of trees (estimators): n_estimators = 100
2. Set maximum features for splitting: max_features = n_features / 3
   - For regression: typically n_features / 3
3. Set random state: random_state = 42 (for reproducibility)
4. Set bootstrap sampling: bootstrap = True
5. Set out-of-bag score: oob_score = True

### Step 3: Build Random Forest Model
1. Initialize RandomForestRegressor with parameters
2. Prepare to build multiple decision trees

### Step 4: Training Phase - For Each Tree to Build

#### Step 4.1: Create Bootstrap Sample
1. Randomly select n samples from training data with replacement
   - Sample size = same as original data set (n)
   - Some samples may appear multiple times
   - Some samples may not appear (out-of-bag samples)

#### Step 4.2: Build Decision Tree
For each decision tree in the forest:

**Start with all bootstrap samples at root node**

**For each node:**

1. **Check stopping criteria:**
   - If node contains < min_samples_split samples, make it a leaf
   - If tree depth ≥ max_depth, make it a leaf
   - If all samples have same target value, make it a leaf
   
2. **Select random feature subset:**
   - Randomly select m features where m < n (m = n/3 for regression)
   - This introduces randomness and decorrelation

3. **Find best split:**
   For each selected feature:
   - Try different split points
   - Calculate MSE (Mean Squared Error) for each split:
     ```
     MSE_left = Σ(yᵢ - ȳ_left)² / n_left
     MSE_right = Σ(yᵢ - ȳ_right)² / n_right
     ```
   - Choose split that minimizes weighted MSE:
     ```
     MSE_total = (n_left/n)·MSE_left + (n_right/n)·MSE_right
     ```

4. **Split the data:**
   - Samples with feature ≤ threshold go to left child
   - Samples with feature > threshold go to right child

5. **Repeat recursively** for left and right children

6. **Create leaf node when stopping criteria met:**
   - Leaf prediction = mean of target values in that node
   - ŷ_leaf = Σyᵢ / n_leaf

### Step 5: Repeat for All Trees
1. Repeat Step 4 for n_estimators (100) times
2. Each tree is built on different bootstrap sample
3. Each split uses different random feature subset
4. Store all trained trees in the forest

### Step 6: Prediction Phase
For each new sample x:

1. **Pass through all trees:**
   For tree t = 1 to n_estimators:
   - Start at root node
   - Follow decision path based on feature values
   - Reach a leaf node
   - Get prediction ŷₜ from that leaf
   
2. **Aggregate predictions:**
   - For regression: compute average
   ```
   ŷ_final = (1/n_estimators) · Σₜ ŷₜ
   ```

3. Return final averaged prediction

### Step 7: Out-of-Bag (OOB) Evaluation
1. For each sample in training data:
   - Identify trees that did NOT use this sample (OOB trees)
   - Get predictions from only these trees
   - Average OOB predictions
   
2. Calculate OOB score:
   - Compare OOB predictions with actual values
   - Provides unbiased estimate of model performance

### Step 8: Feature Importance Calculation
1. For each feature:
   - Track total MSE reduction across all trees
   - Weight by number of samples at each split
   
2. Normalize importance scores:
   ```
   importance_i = Σ(MSE_reduction_i) / Σ(all_MSE_reductions)
   ```

### Step 9: Model Evaluation
1. Make predictions on test set using all trees
2. Calculate R² Score:
   - R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
3. Calculate RMSE:
   - RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
4. Calculate MAE:
   - MAE = Σ|yᵢ - ŷᵢ| / n

### Step 10: Cross-Validation
1. Split data into k=5 folds
2. For each fold i:
   - Train Random Forest on k-1 folds
   - Validate on fold i
   - Record R² score
3. Compute mean CV score and standard deviation

### Step 11: Save Model
1. Serialize trained Random Forest using joblib
2. Save to 'random_forest.joblib'
3. Store performance metrics and feature importances in JSON

### Step 12: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 100 | Number of trees in the forest |
| max_features | n/3 | Features to consider at each split |
| bootstrap | True | Use bootstrap sampling |
| oob_score | True | Calculate out-of-bag score |
| random_state | 42 | Random seed for reproducibility |
| min_samples_split | 2 | Minimum samples to split a node |

## Performance Metrics

- **R² Score**: 0.9341 (93.41% variance explained)
- **RMSE**: 0.0517
- **MAE**: 0.0386
- **Status**: Second best performing model

## Advantages
- Reduces overfitting through ensemble averaging
- Handles non-linear relationships well
- Provides feature importance rankings
- Robust to outliers
- No need for feature scaling
- Implicit feature selection

## Mathematical Formulation

**Bootstrap Sample:**
```
B_t = {(x₁*, y₁*), ..., (xₙ*, yₙ*)}
where each (xᵢ*, yᵢ*) is randomly sampled with replacement
```

**Random Feature Selection:**
```
At each split, randomly select m features where m ≈ n_features / 3
```

**MSE Criterion for Split:**
```
MSE_split = (n_left/n)·MSE_left + (n_right/n)·MSE_right

MSE_node = (1/n_node)·Σᵢ(yᵢ - ȳ_node)²
```

**Final Prediction:**
```
ŷ = (1/T)·Σₜ₌₁ᵀ hₜ(x)

where T = number of trees, hₜ(x) = prediction from tree t
```

**Feature Importance:**
```
Importance(feature_j) = Σ_trees Σ_splits_using_j (MSE_before - MSE_after)
```
