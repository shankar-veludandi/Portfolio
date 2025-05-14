# Predicting Molecular Biodegradability  
**ChemsRUs Codalab Competition Entry**  
Shankar Veludandi (veluds)

---

## Overview  
This project tackles the 2024 ChemsRUs “Predicting Biodegradability” challenge on Codalab.  
We explore logistic regression and SVM classifiers, combined with two feature‐selection methods—RFE and L1‐regularization—to maximize AUC and balanced accuracy on held‐out data.

---

## Data  
- **Features:** 168 molecular descriptors (X0–X167) for 1,055 compounds  
- **Labels:** Biodegradability (1 = biodegradable, –1 = non-biodegradable)  
- **Splits:** 90% train / 10% validation (seed = 200), plus external test set provided by Codalab

---

## Methods  

1. **Baseline models (all 168 features)**  
   - Logistic Regression & SVM (RBF kernel)  
   - Metrics: Balanced Accuracy & AUC  

2. **Feature Selection**  
   - **RFE (Recursive Feature Elimination)** with SVM as estimator → top 40 descriptors  
   - **L1-Regularization (Lasso)** in logistic regression → ~26 nonzero coefficients  

3. **Retrained classifiers** on the reduced feature sets  
   - Evaluate via validation balanced accuracy & AUC  

---

## Key Results  

| Method                              | Dim. | Val Bal Acc | Val AUC  |
|-------------------------------------|------|-------------|----------|
| LR (all features)                   | 168  | 0.838       | 0.925    |
| SVM (all features)                  | 168  | 0.861       | 0.952    |
| **LR + RFE**                        | 40   | 0.905       | 0.961    |
| **SVM + RFE** ★ final entry         | 40   | 0.899       | 0.974    |
| LR + L1 (Lasso)                     | 26   | 0.822       | 0.902    |
| SVM + L1 (Lasso)                    | 26   | 0.142       | 0.912    |

★ **Final submission** to Codalab (AUC 0.9101 on the external test set; feature‐selection bal. acc 0.9603).

---

## How to reproduce  

1. **Install R dependencies**  
   ```r
   install.packages(c("glmnet","e1071","caret","pROC","dplyr","ggplot2","knitr"))
