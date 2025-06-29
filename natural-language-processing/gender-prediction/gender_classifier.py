import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

# Function to convert hexadecimal to RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) if lv == 6 else (0, 0, 0)

# Load the dataset
df = pd.read_csv('gender-classifier-DFE-791531.csv', encoding='latin1')  # Adjust path and encoding as necessary

# DATA PREPROCESSING

# Remove rows with NaN in the 'gender' column
df = df.dropna(subset=['gender'])
# Remove rows with 'unknown' in 'gender' column
df = df[df['gender'] != 'unknown']

# Fill missing textual content with an empty string or drop them
df['text'] = df['text'].fillna('')

# Preprocess 'gender_confidence' (drop rows with NaN values and reset index)
df = df.dropna(subset=['gender:confidence'])
df.reset_index(drop=True, inplace=True)
# Preprocess 'gender_confidence' by filtering out low-confidence rows
confidence_threshold = 0.7
df = df[df['gender:confidence'] >= confidence_threshold]

stop_words = set(stopwords.words('english'))

# Preprocess 'description' (tokenization, stopwords removal, and join tokens)
df['description'] = df['description'].fillna('')  # Fill NaN descriptions with empty string
df['processed_description'] = df['description'].apply(
    lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
)

# Preprocess 'link_color' (convert to RGB and normalize)
df['link_color'] = df['link_color'].apply(lambda x: 'FFFFFF' if len(x) != 6 else x)
df['link_color_rgb'] = df['link_color'].apply(hex_to_rgb)
df[['link_red', 'link_green', 'link_blue']] = pd.DataFrame(df['link_color_rgb'].tolist(), index=df.index) / 255

#FEATURE EXTRACTION

nltk.download('punkt')
nltk.download('stopwords')

df['processed_text'] = df['text'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Join tokens back into strings
df['processed_text_joined'] = df['processed_text'].apply(' '.join)

def hex_to_rgb(value):
    """Converts hexadecimal to RGB."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) if lv == 6 else (0, 0, 0)

# Convert sidebar_color to RGB and then normalize the RGB values
df['sidebar_color_rgb'] = df['sidebar_color'].apply(hex_to_rgb)
df[['red', 'green', 'blue']] = pd.DataFrame(df['sidebar_color_rgb'].tolist(), index=df.index) / 255


# Initialize TF-IDF Vectorizer and transform the processed text
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to 1000 for demonstration
text_features = tfidf_vectorizer.fit_transform(df['processed_text_joined']).toarray()

# Color features
color_features = df[['red', 'green', 'blue']].to_numpy()

# Vectorize 'processed_description' with TF-IDF
description_tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to 1000 for demonstration
description_features = description_tfidf_vectorizer.fit_transform(df['processed_description']).toarray()

# Combine 'text_features', 'color_features', 'description_features', and 'gender_confidence' features
gender_confidence_features = df['gender:confidence'].values.reshape(-1, 1)  # Reshape for concatenation
link_color_features = df[['link_red', 'link_green', 'link_blue']].to_numpy()

# Combine features
X = np.concatenate((text_features, color_features, description_features, gender_confidence_features, link_color_features), axis=1)

# Assuming 'gender' is the target column
y = df['gender'].values

# CLASSIFIER IMPLEMENTATION

# Define a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# DATASET PARTITIONING

# Define the number of folds for k-fold cross-validation
k = 5  # You can choose another value for k

# Define the scoring metrics
scoring = {'accuracy': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_weighted'}

# Perform k-fold cross-validation and collect the results
cv_results = cross_validate(classifier, X, y, cv=k, scoring=scoring, return_train_score=False)

# cv_results is a dictionary where the keys are the metric names prefixed with 'test_'
# and the values are arrays containing the scores for each fold

# PERFORMANCE METRICS CALCULATIONS

# Iterate through each fold and print the scores
for fold_index in range(k):
    print(f"Fold {fold_index+1} - Accuracy: {cv_results['test_accuracy'][fold_index]:.2f}, "
          f"Precision: {cv_results['test_precision'][fold_index]:.2f}, "
          f"Recall: {cv_results['test_recall'][fold_index]:.2f}")

# Calculate the average scores across all folds
average_accuracy = np.mean(cv_results['test_accuracy'])
average_precision = np.mean(cv_results['test_precision'])
average_recall = np.mean(cv_results['test_recall'])

print(f'\nAverage Accuracy: {average_accuracy:.2f}')
print(f'Average Precision: {average_precision:.2f}')
print(f'Average Recall: {average_recall:.2f}')