# Identifying Toxicity Within Esports YouTube Channels

This repository contains the implementation and analysis of a research project aimed at identifying and measuring toxicity in YouTube comments from Indian esports channels. By leveraging the YouTube Data API and machine learning, the study quantifies the level of toxic content and explores its implications for the online esports community.

## Overview

The growing prevalence of online platforms like YouTube has created opportunities for content sharing and interaction. However, these interactions often include toxic comments, potentially impacting the mental well-being of creators and viewers. This project focuses on analyzing the comments on Indian esports YouTube channels to assess their toxicity levels.

### Key Features
- **Data Collection**: Using the YouTube Data API to gather comments from popular Indian esports channels.
- **Machine Learning**: Leveraging logistic regression and Google's Perspective API to classify comments as toxic or non-toxic.
- **Metrics**: Generating performance metrics such as F1 score, precision, and recall to evaluate model accuracy.

---

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Machine Learning Model](#machine-learning-model)
- [Results and Analysis](#results-and-analysis)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Dataset Preparation

### Steps:
1. **Data Collection**: Comments were extracted using the YouTube Data API.
2. **Data Cleaning**:
   - Removed emojis and trailing spaces.
   - Applied stopword removal using the NLTK library.
   - Normalized text to lowercase for consistency.

3. **Labeling**:
   - Used the TextBlob library to calculate sentiment polarity.
   - Labeled comments as positive (1) or negative (-1) based on thresholds.

---

## Data Processing Pipeline

1. **Text Tokenization**: Breaking down comments into words using `CountVectorizer`.
2. **Train-Test Split**: Divided data into training and testing sets.
3. **Feature Engineering**:
   - Extracted textual features such as frequency counts of tokens.

---

## Machine Learning Model

### Model Used:
- Logistic Regression:
  - Used for binary classification (toxic vs. non-toxic).
  - Achieved an F1 score of **0.9209**.

### Evaluation Metrics:
- **Accuracy**: 97.9% on training data, 89.73% on test data.
- **Confusion Matrix**: Analyzed predictions to identify false positives and negatives.

---

## Results and Analysis

1. **Toxicity Distribution**:
   - Toxic comments accounted for **23.16%** on average across channels.
   - Highest toxicity rate observed was **27%**.

2. **Visualization**:
   - Bar charts and graphs illustrating toxicity across channels.

3. **Selected Channels**:
   - Data from the top 5 Indian esports YouTube channels based on subscriber count.

---

## Future Work

- Expanding the dataset to include more channels and videos.
- Incorporating advanced NLP models like transformers to improve accuracy.
- Exploring multilingual datasets to cover non-English comments.

---

## Acknowledgments

We extend our gratitude to:
- **SRM Institute of Science and Technology** for providing the opportunity to work on this project.
- **Dr. Subalalitha C N** and **Dr. Subalalitha C N** for their guidance and support.

---

## References

1. Obadimu, Adewale et al., “Identifying Toxicity Within YouTube Video Comments,” 2019.
2. Joshua Guberman et al., “Quantifying Toxicity on Twitter,” CSCW’16.
3. Kirk R. Williams and Nancy G. Guerra, “Prevalence and Predictors of Internet Bullying,” 2007.
