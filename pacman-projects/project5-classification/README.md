# Project 5: Classification

## Overview
Build and compare three classifiers on digit and face image datasets:

- **Naive Bayes**: Implements a Gaussian/Discrete Naive Bayes classifier.
- **Perceptron**: Linear classifier with online weight updates.
- **MIRA**: Margin-based large-margin online learning algorithm.

## Structure
- `naiveBayes.py`: Model training and smoothing logic.
- `perceptron.py`: Training loop and prediction.
- `mira.py`: MIRA update rule with automatic tuning.
- `dataClassifier.py`: Driver script for training, testing, and comparison.

## Usage Examples
```bash
# Train and test Naive Bayes with tuning
python dataClassifier.py -c naiveBayes --autotune -d digits

# Evaluate Perceptron on faces
python dataClassifier.py -c perceptron -d faces -t 1000 -l 0.01
```
