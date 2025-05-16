# Breast Cancer Classification Pipeline

An end-to-end binary classification workflow using the UCI Breast Cancer dataset to predict malignancy. This project explores decision trees, ensemble methods (Random Forest & XGBoost), statistical comparisons, and hyperparameter tuning.

---

## 1. AI/ML Problem

**Binary classification**: predict whether a tumor is **benign** or **malignant** from 30 numeric features (radius, texture, concavity, etc.).

---

## 2. Approach & Tasks

### Task 1: Decision-Tree Complexity  
- **Goal:** Understand how tree depth, split-size, and leaf-size affect under/overfitting.  
- **Method:** Train `DecisionTreeClassifier` with 3 hyperparameter settings (`max_depth=3`, `min_samples_split=10`, `min_samples_leaf=20`), visualize each tree.  
- **Feature Analysis:**  
  - **Feature importances** identified “mean concave points” & “worst concave points” as top predictors.  
  - **Partial-dependence plots** showed how malignancy probability shifts with those features.  

### Task 2: Ensemble Comparison  
- **Goal:** Compare `RandomForestClassifier` vs `XGBClassifier` on balanced accuracy, precision, recall, and F1.  
- **Method:** 5-, 10-, and 15-fold stratified cross-validation using accuracy, precision, recall, F1.  
- **Results:**  
  | Model        | 5-Fold Acc | 10-Fold Acc | 15-Fold Acc | 5-Fold F1 | 10-Fold F1 | 15-Fold F1 |
  |-------------:|-----------:|------------:|------------:|----------:|-----------:|-----------:|
  | **Random Forest** | 95.81%     | 96.51%      | 95.76%      | 0.9581    | 0.9651     | 0.9576     |
  | **XGBoost**      | 95.68%     | 96.31%      | 96.75%      | 0.9568    | 0.9631     | 0.9674     |  
- **Interpretation:** XGBoost slightly outperforms RF as folds increase; both achieve > 95% accuracy and F1.

### Task 3: Confusion & Statistical Significance  
- **Goal:** Visualize per-fold confusion and test whether RF vs XGB performance differs significantly.  
- **Method:**  
  - Plot confusion matrices for fold 1 (Decision Tree vs RF vs XGB).  
  - Perform paired t-tests on accuracy across folds.  
- **Findings:**  
  - RF vs XGB: no significant difference (p ≈ 0.90).  
  - Both far outperform a single Decision Tree (p < 0.01).  

### Task 4: XGBoost Hyperparameter Tuning  
- **Goal:** Optimize XGBoost’s `learning_rate`, `max_depth`, and `subsample`.  
- **Method:** `GridSearchCV` over learning rates [0.01, 0.1, 0.2], depths [3, 5, 7], subsamples [0.8, 1.0], 5-fold CV.  
- **Best Parameters:**  
  - `learning_rate=0.2`, `max_depth=5`, `subsample=0.8`  
  - **Mean CV accuracy:** 97.36%  
- **Visuals:** Plots showing how accuracy improves with each hyperparameter.

---

## 3. Results

- **Decision Tree** (depth=3): test accuracy ~90–92% but limited by bias/variance trade-off.  
- **Random Forest**: ≈ 96.5% accuracy, 0.965 F1.  
- **XGBoost**: ≈ 96.8% accuracy, 0.967 F1 — best ensemble.  
- **Optimized XGBoost**: 97.4% CV accuracy with tuned hyperparameters.
