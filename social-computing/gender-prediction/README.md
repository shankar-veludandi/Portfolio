## 📁 Gender Prediction from Twitter Profiles

**Problem:** Given a user’s tweets, profile description, and UI color choices, predict the user’s gender.

**Approach & Features**  
- **Textual Features:** TF–IDF vectors over unigrams from tweets & profile descriptions (top 1,000 terms).  
- **Color Features:** Normalized RGB values extracted from sidebar & link color hex codes.  
- **Confidence Feature:** Provided “gender:confidence” score weighted in the model.

**Model**  
- **Classifier:** Multinomial Naive Bayes (scikit-learn’s `MultinomialNB`).

**Evaluation**  
- **Method:** 5-fold cross-validation  
- **Metrics:** Accuracy, weighted Precision & Recall  
- **Results (average):**  
  - Accuracy: 65%  
  - Precision: 65%  
  - Recall: 65%
