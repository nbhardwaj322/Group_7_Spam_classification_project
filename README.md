Spam Classification Using Machine Learning
This project is a part of the AAI 501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

Project Status: Completed

git clone https://github.com/nbhardwaj322/Group_7_Spam_classification_project
cd spam-classification
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt

Project Intro / Objective
The purpose of this project is to develop a machine learning-based SMS spam detection system capable of accurately classifying incoming messages as spam or ham (non-spam).
The system combines Natural Language Processing (NLP) techniques with multiple machine learning models, ultimately identifying XGBoost as the top-performing model.

The project aims to:

Minimize false negatives to protect users from spam threats.

Handle noisy and imbalanced real-world text data.

Evaluate and compare multiple algorithms for optimal performance.

Partner(s) / Contributor(s)
Team Members:

Nitish Bhardwaj

Vijay Agarwal

Bashir Ali

Methods Used
Natural Language Processing (NLP)

Machine Learning

Model Evaluation Metrics (Accuracy, Precision, Recall, F1-score)

Dimensionality Reduction (SVD, PCA)

Handling Imbalanced Data (SMOTE)

Data Visualization

Technologies
Python 3.x

pandas, numpy, scikit-learn

XGBoost

imbalanced-learn

NLTK

matplotlib, seaborn

joblib

Project Description
Dataset
Name: SMS Spam Collection Dataset (UCI Machine Learning Repository)

Size: 5,574 labeled SMS messages

Classes:

Ham (legitimate)

Spam (unwanted/malicious)

Imbalance: 87% ham, 13% spam

Ham messages are generally shorter (<100 characters), while spam messages are often longer and contain promotional or fraudulent content.

Approach
Data Preprocessing: Lowercasing, punctuation removal, tokenization, stopword removal, stemming/lemmatization, label encoding.

Feature Engineering:

TF-IDF vectorization

Dimensionality reduction with TruncatedSVD and PCA

KMeans clustering features

Message length metadata

Class Imbalance Handling: SMOTE oversampling for minority spam class.

Model Building: Logistic Regression, Linear SVC, RandomForest, KNN, GradientBoosting, XGBoost.

Results:
o	XGBoost achieves the highest F1 score (~0.93), indicating a strong balance between precision and recall, making it the most reliable choice overall.

Evaluation: Accuracy, Precision, Recall, F1-score, ROC curve, Feature importance.

