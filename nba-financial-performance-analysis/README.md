# Analysis of NBA Franchise Financial Decisions on On-Court Performance

End-to-end analysis from data scraping and cleaning to modeling and evaluation.

## 1. Trade Market Dynamics & Team Success

**Objective**  
Investigate whether an NBA franchises’ financial decision making (salary‐cap space, luxury‐tax exemptions, cash, and contract value) predicts next‐season on‐court performance and playoff qualification.

**Data**  
- **Seasons:** 2012–13 through 2023–24 (10 years)  
- **Sources:**  
  - NBA advanced metrics via the NBA API (`nba_api`)  
  - Salary & luxury‐tax data scraped from Spotrac  
- **Panel:** 360 team‐season observations after cleaning & merging

**Methods**  
1. **Feature Engineering:**  
   - Five utilization ratios (cap, tax, cash, AAV, off-season spend) standardized and lagged by one year  
   - Composite performance index (PC1) via PCA on five efficiency metrics  
2. **Models:**  
   - **Regression:** Pooled OLS, team fixed‐effects, LASSO, Random Forest  
   - **Classification:** Logistic Regression & Random Forest (playoff prediction)  
3. **Evaluation:**  
   - **Regression:** Out-of-sample RMSE (~1.7–1.8) & R²  
   - **Classification:** ROC‐AUC (~0.49, no better than chance)  

**Key Findings**  
- Only **cap_util** and **cash_util** survived LASSO penalization  
- All regressors yielded similar RMSE (~1.70)  
- Playoff classifiers failed to discriminate (AUC ≈ 0.49)  
- Financial “savvy” alone is a weak predictor of team success
