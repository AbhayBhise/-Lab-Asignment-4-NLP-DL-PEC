# Lab Assignment 4 — NLP Preprocessing and Text Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![Dataset](https://img.shields.io/badge/Dataset-Twitter%20Airline%20Sentiment-1DA1F2?logo=twitter)

---

## Overview

This assignment implements an end-to-end NLP pipeline for **sentiment-based text classification** on real-world Twitter data. Tweets from US airline passengers are classified into three sentiment categories — **Positive**, **Neutral**, and **Negative** — using classical machine learning techniques.

The entire workflow covers text cleaning, NLP preprocessing, feature extraction, model training, and performance evaluation — fulfilling all objectives of Lab Assignment 4.

---

## Dataset

**Twitter US Airline Sentiment**  
Source: [Kaggle — crowdflower/twitter-airline-sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

| Property | Details |
|----------|---------|
| Total Samples | ~14,600 tweets |
| Classes | Positive, Neutral, Negative |
| Class Distribution | ~63% Negative, ~21% Neutral, ~16% Positive |
| Domain | Passenger tweets about 6 US airlines |

The dataset was chosen because it represents real, noisy, informal text — making it a practical and challenging classification task compared to clean news corpora.

---

## Project Structure

```
├── Lab_Assignment_4_NLP.ipynb     # Main notebook (run this)
├── README.md                      # This file
```

Output files generated after running the notebook:

```
├── eda_plots.png                  # Class distribution + tweet length charts
├── wordclouds.png                 # Word clouds per sentiment class
├── model_comparison.png           # Accuracy + training time bar charts
├── confusion_matrices.png         # Confusion matrix for all 6 models
├── tfidf_vs_countvec.png          # Grouped comparison chart
```

---

## NLP Preprocessing Pipeline

Raw tweets contain a lot of noise — mentions, hashtags, URLs, slang. The following steps were applied to clean and normalize the text before feeding it into any model:

| Step | Technique | What it does |
|------|-----------|--------------|
| 1 | Lowercasing | Standardizes text, reduces vocabulary size |
| 2 | Remove URLs | Strips `http://...` links |
| 3 | Remove @mentions & #hashtags | Removes Twitter-specific tokens |
| 4 | Remove special characters & digits | Keeps only alphabetic content |
| 5 | Tokenization | Splits sentences into individual words |
| 6 | Stopword Removal | Drops words like "the", "is", "at" |
| 7 | Stemming | Reduces words to root form (`running → run`) |
| 8 | Lemmatization | Context-aware normalization (`better → good`) |

After preprocessing, average token count per tweet dropped noticeably, confirming noise was successfully removed.

---

## Text Vectorization

Two vectorization strategies were implemented and compared:

- **TF-IDF Vectorizer** — weights words by importance across the corpus (preferred for classification)
- **CountVectorizer** — raw word frequency counts (Bag of Words baseline)

Both use `ngram_range=(1,2)` to capture single words and two-word phrases like *"flight delayed"* or *"great service"*, with a maximum of 30,000 features.

---

## Models Trained

Three classifiers were trained on both vectorizers (6 combinations total):

| Classifier | Why chosen |
|-----------|------------|
| Multinomial Naive Bayes | Fast probabilistic baseline |
| Linear SVM | Strong performer for high-dimensional text |
| Random Forest | Ensemble method, robust to noisy data |

---

## Results

| Rank | Model + Vectorizer | Accuracy | Train Time |
|------|--------------------|----------|------------|
| 1 | Linear SVM + TF-IDF | **77.73%** | 0.29s |
| 2 | Random Forest + TF-IDF | 77.05% | 43.96s |
| 3 | Random Forest + CountVec | 76.57% | 27.42s |
| 4 | Linear SVM + CountVec | 76.30% | 0.63s |
| 5 | Naive Bayes + CountVec | 76.26% | 0.00s |
| 6 | Naive Bayes + TF-IDF | 73.05% | 0.01s |

**Linear SVM + TF-IDF** achieved the best accuracy at **77.73%**.

One interesting observation — Naive Bayes performed better with CountVectorizer than TF-IDF. This is expected behavior: Naive Bayes internally assumes integer-like frequency counts, so raw CountVec features suit it better than floating-point TF-IDF weights.

### Why ~77% accuracy?

The accuracy is moderate for a reason. Twitter data is inherently difficult to classify:
- Heavy class imbalance (63% of tweets are negative)
- Sarcasm and informal language that bag-of-words models cannot handle
- Short tweet length limits the amount of textual signal available

Even transformer-based models like BERT typically reach 85–88% on this dataset. For classical ML with simple vectorization, 77% is a realistic and honest result.

---

## How to Run

### 1. Open in Google Colab
Upload `Lab_Assignment_4_NLP.ipynb` to [Google Colab](https://colab.research.google.com/) — no GPU needed, CPU runtime is sufficient.

### 2. Set up Kaggle API
- Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
- This downloads `kaggle.json` to your machine
- Upload it when prompted in Step 1 of the notebook

### 3. Run All Cells
Execute cells from top to bottom. The notebook will automatically:
1. Download and load the dataset
2. Run the full preprocessing pipeline
3. Vectorize text with TF-IDF and CountVectorizer
4. Train and evaluate all 6 model combinations
5. Generate and save all plots
6. Run predictions on 3 custom tweets

---

## Sample Custom Predictions

At the end of the notebook, the best model (Linear SVM + TF-IDF) is tested on hand-written tweets:

| Tweet | Predicted Sentiment |
|-------|-------------------|
| "Thank you @united for the amazing service, truly enjoyed the flight!" | POSITIVE 😊 |
| "Flight delayed again, no one at the gate, absolutely terrible experience." | NEGATIVE 😡 |
| "Just landed at JFK, average flight nothing special." | NEGATIVE 😡 |

The third tweet was predicted as Negative instead of Neutral — a minor misclassification likely caused by the model's bias toward the dominant negative class in the training data.

---

## Learning Outcomes Covered

- Applied NLP preprocessing — tokenization, stopword removal, stemming, lemmatization
- Implemented TF-IDF and CountVectorizer for text vectorization
- Built and compared multiple ML classification models
- Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix

---

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly. Available: https://www.nltk.org/book/
- Scikit-learn documentation — [scikit-learn.org](https://scikit-learn.org)
- Dataset: Crowdflower (2015). Twitter US Airline Sentiment. Kaggle.

---

**Name:** Abhay Bhise 
**PRN:** 202402040016
**Subject:** Deep Learning (PEC)
**Assignment:** Lab Assignment 4
