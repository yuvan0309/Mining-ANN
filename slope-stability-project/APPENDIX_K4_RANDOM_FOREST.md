# APPENDIX K4

## ALGORITHM USED FOR RANDOM FOREST REGRESSION

### Algorithm: Random Forest Regressor with Bootstrap Aggregating

---

## Step 1: Load and Prepare Data

1. Load the training dataset containing material properties and target FoS values
2. Separate features (X) from target variable (y)
3. Split data into training and test sets (80:20 ratio, 288 training, 73 test)

---

## Step 2: Initialize Hyperparameters

1. **Number of estimators**: n_estimators = 100
   - Number of trees in the forest

2. **Maximum depth**: max_depth = None
   - Trees grow until leaves are pure or have min_samples_split samples

3. **Minimum samples split**: min_samples_split = 2
   - Minimum samples required to split internal node

4. **Minimum samples leaf**: min_samples_leaf = 1
   - Minimum samples required at leaf node

5. **Maximum features**: max_features = 'sqrt'
   - Number of features for best split: √(n_features) = √4 = 2

6. **Bootstrap**: bootstrap = True
   - Use bootstrap sampling for training

7. **Out-of-Bag score**: oob_score = False
   - Don't compute OOB score

8. **Random state**: random_state = 42
   - For reproducibility

9. **n_jobs**: n_jobs = -1
   - Use all CPU cores for parallel processing

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

## Step 4: Build Random Forest Model

1. Initialize RandomForestRegressor with parameters
2. Create pipeline combining scaler and regressor
3. Set up parallel tree building
4. Configure bootstrap sampling

---

## Step 5: Training Phase - Ensemble Learning

**For each tree t = 1 to 100:**

1. **Bootstrap Sampling (Bagging):**
   ```
   Create bootstrap sample Dₜ:
   - Sample n samples with replacement from training set
   - Expected: ~63.2% unique samples, ~36.8% duplicates
   - Out-of-bag (OOB) samples: remaining ~36.8%
   ```

2. **Feature Randomization:**
   ```
   At each node split:
   - Randomly select m = √d features (m = 2 from 4 features)
   - Consider only these m features for splitting
   - Increases tree diversity
   ```

3. **Build Decision Tree on Dₜ:**
   ```
   Start with root node containing all bootstrap samples
   
   For each node:
       a. Stop if pure or min_samples_split not met
       b. Randomly select m = 2 features
       c. Find best split among these m features
       d. Split node into left and right children
       e. Recursively build subtrees
   ```

4. **Split Criterion (MSE Reduction):**
   ```
   For each candidate split (feature j, threshold s):
   
   MSE_parent = Σᵢ(yᵢ - ȳ_parent)² / n_parent
   
   MSE_left = Σᵢ∈left(yᵢ - ȳ_left)² / n_left
   MSE_right = Σᵢ∈right(yᵢ - ȳ_right)² / n_right
   
   Impurity_reduction = MSE_parent - (n_left/n × MSE_left + n_right/n × MSE_right)
   
   Best split = argmax(Impurity_reduction)
   ```

5. **Leaf Node Prediction:**
   ```
   prediction at leaf = mean(target values in leaf)
   ŷ_leaf = Σyᵢ / n_leaf
   ```

6. **Stopping Criteria:**
   - All samples in node have same target value
   - Node has fewer than min_samples_split samples
   - Maximum depth reached (no limit in this case)
   - No further impurity reduction possible

---

## Step 6: Tree Building Details

**Recursive Tree Construction:**

```
function BuildTree(data, depth):
    if stopping_criteria_met:
        return LeafNode(mean(targets))
    
    # Random feature selection
    candidate_features = random_sample(all_features, m=2)
    
    # Find best split
    best_gain = -∞
    best_split = None
    
    for feature in candidate_features:
        for threshold in unique_values(feature):
            gain = calculate_gain(feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold)
    
    if best_gain <= 0:
        return LeafNode(mean(targets))
    
    # Split data
    left_data = data[feature <= threshold]
    right_data = data[feature > threshold]
    
    # Recursive calls
    left_child = BuildTree(left_data, depth+1)
    right_child = BuildTree(right_data, depth+1)
    
    return InternalNode(feature, threshold, left_child, right_child)
```

---

## Step 7: Prediction Phase

**For each new sample x:**

1. **Scale input features:**
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Get prediction from each tree:**
   ```
   For t = 1 to 100:
       Start at root of tree t
       While current node is not leaf:
           If x[split_feature] <= split_threshold:
               current_node = left_child
           Else:
               current_node = right_child
       predictions[t] = leaf_value
   ```

3. **Aggregate predictions (averaging):**
   ```
   ŷ = (1/T) · Σₜ₌₁ᵀ predictions[t]
   
   Where T = 100 trees
   ```

4. **Return final FoS prediction**

---

## Step 8: Model Evaluation

**Test Set Performance:**
1. Calculate R² Score:
   ```
   R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
   R² = 0.9356 (93.56%)
   ```

2. Calculate RMSE:
   ```
   RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
   RMSE = 0.0883
   ```

3. Calculate MAE:
   ```
   MAE = Σ|yᵢ - ŷᵢ| / n
   MAE = 0.0613
   ```

**Training Set Performance:**
```
R²_train = 0.9787 (97.87%)
```

**Generalization:**
```
Overfitting Gap = R²_train - R²_test = 4.31%
```

---

## Step 9: Feature Importance Analysis

**Calculate feature importance using impurity reduction:**

```
For each feature j:
    importance[j] = 0
    For each tree t in forest:
        For each node n in tree t that splits on feature j:
            importance[j] += (n_samples[n]/N) × impurity_reduction[n]
    importance[j] /= n_trees

Normalize: importance = importance / sum(importance)
```

**Feature Rankings:**
1. Friction Angle: ~38%
2. Cohesion: ~34%
3. Unit Weight: ~18%
4. Ru: ~10%

---

## Step 10: Variance Analysis

**Prediction variance across trees:**

```
For each test sample:
    variance = (1/T) · Σₜ(prediction_t - mean_prediction)²
    
Average variance provides model uncertainty estimate
```

---

## Step 11: Cross-Validation

1. Perform 5-fold cross-validation
2. For each fold:
   - Train on 4 folds
   - Validate on 1 fold
   - Record R² score
3. Calculate mean and std of CV scores

---

## Step 12: Save Model

1. Serialize trained pipeline using joblib
2. Save to 'best_model_random_forest.pkl'
3. Save scaler separately
4. Store feature importance values
5. Record tree statistics

---

## Step 13: End

---

## Key Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 100 | Number of trees in forest |
| max_depth | None | No depth limit (grow until pure) |
| min_samples_split | 2 | Minimum samples to split node |
| min_samples_leaf | 1 | Minimum samples at leaf |
| max_features | 'sqrt' | Features per split: √4 = 2 |
| bootstrap | True | Use bootstrap sampling |
| oob_score | False | No OOB scoring |
| n_jobs | -1 | Use all CPU cores |
| random_state | 42 | Reproducibility seed |

---

## Performance Metrics

- **R² Score (Test)**: 0.9356 (93.56% variance explained)
- **R² Score (Training)**: 0.9787 (97.87% variance explained)
- **RMSE**: 0.0883
- **MAE**: 0.0613
- **Overfitting Gap**: 4.31%
- **Status**: Third best performing model
- **Training Time**: Fast (parallel tree building)

---

## Advantages

1. **Robust to Overfitting**: Ensemble of diverse trees
2. **Parallel Processing**: Fast training with multiple cores
3. **Feature Importance**: Clear interpretability
4. **No Feature Scaling Required**: Tree-based (but we scale anyway)
5. **Handles Non-linearity**: Captures complex patterns
6. **Low Variance**: Averaging reduces prediction variance
7. **Minimal Hyperparameter Tuning**: Good default performance

---

## Disadvantages

1. **Lower Accuracy**: 93.56% vs 94.26% (Gradient Boosting)
2. **Memory Usage**: Stores 100 full-depth trees
3. **Black Box**: Less interpretable than single tree
4. **Prediction Time**: Slower than single model (100 trees)
5. **Not as Powerful**: Sequential boosting often better than bagging

---

## Mathematical Formulation

### Ensemble Prediction:
```
F(x) = (1/T) · Σₜ₌₁ᵀ fₜ(x)

Where:
- T = 100 trees
- fₜ(x) = prediction from tree t
- Simple averaging aggregation
```

### Bootstrap Sample:
```
Dₜ = {(x₁*, y₁*), (x₂*, y₂*), ..., (xₙ*, yₙ*)}

Where each (xᵢ*, yᵢ*) is sampled uniformly with replacement from D
```

### Split Criterion (MSE):
```
MSE(S) = (1/|S|) · Σᵢ∈S (yᵢ - ȳₛ)²

Where:
S = set of samples in node
ȳₛ = mean of targets in S
```

### Impurity Reduction:
```
ΔImpurity = MSE(S_parent) - [p_left·MSE(S_left) + p_right·MSE(S_right)]

Where:
p_left = |S_left| / |S_parent|
p_right = |S_right| / |S_parent|
```

### Feature Importance:
```
Importance(feature_j) = Σ_trees Σ_nodes (weighted_impurity_reduction)

Normalized:
Importance(feature_j) = Importance(feature_j) / Σⱼ Importance(feature_j)
```

---

## Algorithm Complexity

- **Training Time**: O(n · log(n) · d · m · T)
  - n = samples (288)
  - d = features (4)
  - m = features per split (2)
  - T = trees (100)
  - Parallelized across trees

- **Prediction Time**: O(T · log(n))
  - T = trees (100)
  - log(n) = average tree depth
  - Can be parallelized

- **Space Complexity**: O(T · n · log(n))
  - Stores all trees (can be large)

---

## Bootstrap Aggregating (Bagging)

**Key Concept:** Train multiple models on different subsets and average predictions

**Benefits:**
1. **Variance Reduction:**
   ```
   Var(F) = Var((1/T)·Σfₜ) = (1/T)·Var(f)  [if uncorrelated]
   ```

2. **Out-of-Bag (OOB) Samples:**
   ```
   Probability of sample not selected: (1 - 1/n)ⁿ ≈ 0.368
   ~36.8% of samples are OOB for each tree
   ```

3. **Diversity through randomness:**
   - Bootstrap sampling (row-wise diversity)
   - Feature randomization (column-wise diversity)

---

## Random Feature Selection

**max_features = 'sqrt':**

```
At each split, consider m = √d = √4 = 2 random features

This means:
- Out of 4 features, only 2 are candidates for each split
- Different trees likely split on different features
- Increases tree diversity
- Reduces correlation between trees
```

**Impact on Variance:**
```
If trees perfectly correlated: Var(F) = Var(f)
If trees uncorrelated: Var(F) = Var(f) / T

Random feature selection reduces correlation
```

---

## Bias-Variance Tradeoff

Random Forest characteristics:

1. **Individual Trees (High Variance, Low Bias):**
   - Deep trees overfit
   - High variance in predictions

2. **Ensemble (Low Variance, Low Bias):**
   - Averaging reduces variance
   - Maintains low bias of deep trees
   - Sweet spot in bias-variance tradeoff

**Mathematical:**
```
MSE = Bias² + Variance + Irreducible Error

Random Forest:
- Bias ≈ Bias(single deep tree) [low]
- Variance ≈ Variance(single tree) / T [reduced]
```

---

## Comparison with Boosting

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training | Parallel | Sequential |
| Trees | Independent | Dependent |
| Combination | Averaging | Weighted sum |
| Focus | Variance reduction | Bias reduction |
| Overfitting | Less prone | More prone |
| Test R² | 93.56% | 94.26% |
| Speed | Fast | Slower |

---

## Conclusion

Random Forest achieved solid performance (R² = 93.56%) with good generalization (4.31% overfitting gap). Its parallel training, robustness, and interpretability make it a reliable choice for slope stability prediction. While slightly less accurate than boosting methods, its simplicity and stability are valuable for production systems requiring model transparency and fast training.
