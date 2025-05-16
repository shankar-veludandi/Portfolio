# Movie Rating Predictor (Neural Network vs. Linear Regression)

An end-to-end regression pipeline predicting user–movie star ratings on MovieLens data, comparing a simple two-layer neural network against a linear‐regression baseline.

---

## 1. AI/ML Problem

**Regression:** Predict each user’s 1–5 rating for a given movie based solely on **UserID** and **MovieID** features.

---

## 2. Approach

1. **Data & Preprocessing**  
   - Loaded 1,000,209 MovieLens ratings (`UserID`, `MovieID`, `Rating`, `Timestamp`).  
   - Sampled 10,000 records for efficiency.  
   - Normalized `UserID` and `MovieID` to [0,1].  
   - Split into Train (70%), Dev (15%), Test (15%).

2. **Neural Network Model**  
   - Architecture: two fully-connected layers (2→10→1) with ReLU in the hidden layer.  
   - Loss: Mean Squared Error (MSE).  
   - Optimizer: Adam (lr=0.01), trained for 1 000 epochs with batch size 32.  

   ![](notebooks/figures/train_dev_loss.png)

3. **Baseline Linear Regression**  
   - Fitted `sklearn.linear_model.LinearRegression` on the same features.  
   - Evaluated test‐set MSE.

---

## 3. Results

| Model                       | Test MSE   |
|-----------------------------|-----------:|
| **Two-Layer Neural Network**| **1.1932** |
| **Linear Regression**       | 1.2022     |

> The neural network converged smoothly (train/dev loss ≈1.21 after ~200 epochs) and modestly outperformed linear regression, highlighting that even simple non-linear models can capture subtle user–movie interactions beyond a purely linear fit.
