# Gender Prediction from Twitter Profiles

## Problem Statement  
Given a Twitter user’s recent tweets, profile description, sidebar & link color choices, and provided gender‐confidence score, predict the user’s gender as **male** or **female**. :contentReference[oaicite:8]{index=8}

---

## Approach & Features  

1. **Textual Features**  
   - TF–IDF vectors (top 1,000 unigrams) over tweet text (`processed_text_joined`) and profile descriptions (`processed_description`). :contentReference[oaicite:9]{index=9}  
2. **Color Features**  
   - Normalized RGB values extracted from `sidebar_color` and `link_color`. :contentReference[oaicite:10]{index=10}  
3. **Confidence Feature**  
   - Provided `gender:confidence` score (filtered ≥ 0.7). :contentReference[oaicite:11]{index=11}  

**Classifier:** Multinomial Naive Bayes (`sklearn.naive_bayes.MultinomialNB`) :contentReference[oaicite:12]{index=12}  
**Evaluation:** 5-fold cross-validation with accuracy, weighted precision, and weighted recall :contentReference[oaicite:13]{index=13}

---

## Results  

| Fold | Accuracy | Precision | Recall |
|-----:|---------:|----------:|-------:|
| 1    | 0.64     | 0.64      | 0.64   |
| 2    | 0.66     | 0.65      | 0.66   |
| 3    | 0.67     | 0.67      | 0.67   |
| 4    | 0.65     | 0.66      | 0.65   |
| 5    | 0.62     | 0.64      | 0.62   |
| **Average** | **0.65** | **0.65** | **0.65** |

Overall, the model achieves **65 %** accuracy (and matching weighted precision/recall), indicating moderate success at distinguishing gender from stylistic and linguistic cues. :contentReference[oaicite:14]{index=14}

---

## Usage  

1. **Clone & enter folder:**
   ```bash
   git clone https://github.com/shankar-veludandi/Portfolio.git
   cd Portfolio/social-computing/gender-prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy nltk scikit-learn
   python -m nltk.downloader punkt stopwords
   ```
3. **Run the classifier:** `python gender_classifier.py`
