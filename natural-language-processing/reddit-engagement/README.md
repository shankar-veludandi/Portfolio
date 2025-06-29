# Engagement Classification in r/AskReddit

## Problem Statement  
Classify r/AskReddit posts as **high engagement** or **low engagement** based on linguistic and temporal cues in the post title.

- **High engagement**: posts with comment counts above the dataset median  
- **Low engagement**: posts with comment counts at or below the median

---

## Data  
- **Source:** 500 “hot” posts scraped via PRAW (Python Reddit API Wrapper)  
- **Total samples:** 13,315 post titles with corresponding comment counts and timestamps  
  - **Training set:** 10,652 titles  
  - **Test set:** 2,663 titles  
- **Preprocessing:**  
  - Whitespace & URL removal, lowercasing, tokenization, stop-word filtering  
  - Alphanumeric filtering to remove punctuation

---

## Approach

**Feature Engineering**

1. **Text Features**  
   - TF–IDF vectors over unigrams & bigrams  
   - Part-of-Speech tag counts  
   - Readability metrics (average sentence length, Flesch score)  
2. **Sentiment**  
   - NLTK’s `SentimentIntensityAnalyzer` compound score  
3. **Temporal**  
   - Hour-of-day bucket (morning/afternoon/evening/night)  
   - Day-of-week indicator  

**Model:**  
- Support Vector Machine (`sklearn.svm.SVC`) with RBF kernel  
- Hyperparameters (`C`, `gamma`) tuned via `GridSearchCV`

---

## Results  

| Class              | Accuracy | Precision | Recall | F1-Score |
|--------------------|---------:|----------:|-------:|---------:|
| **Low engagement** |  82 %    |   0.83    |  0.92  |   0.87   |
| **High engagement**|  82 %    |   0.80    |  0.64  |   0.71   |

**Confusion Matrix** (on test set of 2,663 posts):  
- True Negatives (low→low): 527  
- False Positives (low→high):  48  
- False Negatives (high→low): 111  
- True Positives (high→high): 197  

> **Key insight:** The model is highly effective at identifying low-engagement posts (high recall), with more room to improve recall on high-engagement cases.

---

## Usage  

1. **Clone the repo & enter the folder:**  
   ```bash
   git clone https://github.com/shankar-veludandi/Portfolio.git
   cd Portfolio/social-computing/reddit-engagement
   ```
2. **Install dependencies:**
   ```bash
   pip install praw nltk spacy pandas numpy scikit-learn matplotlib seaborn
   python -m spacy download en_core_web_sm
   ```
3. **Run the classifier:** `python reddit_engagement.py`

   
