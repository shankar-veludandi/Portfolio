## 📁 Engagement Classification in r/AskReddit

**Problem:** Classify Reddit posts as “high engagement” or “low engagement” based on title phrasing, sentiment, and timing.

**Approach & Features**  
- **Text Features:** TF–IDF over unigrams & bigrams; Part-of-Speech tag counts; sentence complexity metrics.  
- **Sentiment:** NLTK’s Sentiment Intensity Analyzer scores for each title.  
- **Temporal:** Hour-of-day and day-of-week descriptors.  

**Model**  
- **Classifier:** Support Vector Machine (scikit-learn’s `SVC`) with hyperparameter tuning via `GridSearchCV`.

**Evaluation**  
- **Method:** 80/20 train/test split  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- **Results:**  
  - Overall Accuracy: 82%  
  - Class 0 (low engagement) F1: 0.87  
  - Class 1 (high engagement) F1: 0.71  
