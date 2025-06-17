# Customer Churn Modeling (Logistic Regression from Scratch)

A from-scratch logistic‐regression pipeline tackling binary churn prediction on the Telco Customer Churn dataset. This project spans theoretical derivation, exploratory data analysis, feature engineering, and an in-depth comparison of gradient-descent variants and advanced optimizers.


---

## 1. Problem Statement

**Predict whether a customer will churn (yes/no) based on demographic, service, and billing features** using logistic regression learned via gradient descent.

---

## 2. Approach

1. **Objective Derivation**  
   – Formulated the logistic‐regression likelihood, log‐likelihood, and Negative Log‐Likelihood (NLL) as the optimization target.  
   – Discussed MLE vs. MAP and logistic‐regression assumptions (IID, linear log-odds).  

2. **Exploratory Data Analysis**  
   – **Dataset:** 7,043 customers, 24 predictors.  
   – **Churn rate:** ~26.5%.  
   – **Key insights:**  
     - Month-to-month contracts churn ~45% vs. ~5% for two-year contracts.  
     - High TotalCharges & short tenure correlate strongly with churn.  
   – **Feature selection:**  
     - Dropped multi-collinear columns via VIF (>10).  
     - Hierarchical clustering & heatmap revealed natural feature groupings.

     

3. **Model Implementation**  
   – Built `LogisticRegressionScratch` supporting **batch**, **stochastic**, and **mini-batch** gradient descent.  
   – Extended to four optimizers: **SGD**, **Momentum**, **RMSProp**, and **Adam**.

---

## 3. Results

### 3.1 Gradient‐Descent Variants

| Method       | Final Cost | Test Accuracy |
|-------------:|-----------:|--------------:|
| **Batch**    | 0.6004     | 68.77%        |
| **Stochastic** | 0.5865   | 68.56%        |
| **Mini-Batch** | 0.5802   | 69.34%        |

Mini-batch achieves the lowest cost and best accuracy among pure GD variants.

### 3.2 Advanced Optimizers

| Optimizer | Test Accuracy | Precision | Recall | F1-Score |
|----------:|--------------:|----------:|-------:|---------:|
| **SGD**     | 68.77%       | 45.30%    | 85.03% | 59.11%   |
| **Momentum**| 69.06%       | 45.57%    | 85.29% | 59.40%   |
| **RMSProp** | 68.99%       | 45.53%    | 85.83% | 59.50%   |
| **Adam**    | 68.99%       | 45.49%    | 85.03% | 59.27%   |

RMSProp yields the highest F1 (59.50%), while Momentum converges more smoothly than vanilla SGD.

#### Hyperparameter Tuning (Momentum)

- **Best F1:** 59.89%  
- **Best (learning_rate, βₘ):** (0.001, 0.95)  


