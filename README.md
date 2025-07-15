## 📩 Ham vs Spam Email Classification – NLP & Machine Learning
# 📌 Overview
This project implements a binary text classification system to detect whether a given message is ham (legitimate) or spam using Natural Language Processing (NLP) techniques and classical machine learning algorithms.

Using a labeled dataset of real-world messages, the model applies preprocessing, tokenization, vectorization, and classification to effectively flag spam messages.

# 🧠 Problem Statement
Email spam is one of the most common nuisances for users and businesses. Automating spam detection using NLP helps:

Protect users from phishing and junk

Improve email deliverability

Reduce human workload in filtering emails

🛠️ Tools & Technologies
Programming Language: Python

Libraries: pandas, nltk, scikit-learn

ML Models: Naive Bayes, Logistic Regression, etc.

NLP Techniques: Tokenization, Stopword Removal, Stemming, TF-IDF Vectorization

# 🗃️ Dataset
File: spam_ham_dataset.csv

Features:

label: Indicates if the message is "spam" or "ham"

text: Raw message content

Source: spam_ham_dataset.csv

# 🔄 Preprocessing Pipeline
Tokenization using nltk.word_tokenize()

Stopword removal and punctuation filtering

Stemming using PorterStemmer

TF-IDF Vectorization to convert text into numerical features

Train-Test Split using train_test_split()

# 🤖 Model Training
Several classifiers were explored including:

Multinomial Naive Bayes

Logistic Regression

Model evaluation was done using accuracy, precision, recall, and confusion matrix.

