#!/usr/bin/env python3

import pandas as pd
import re
import string
import joblib
import warnings
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
np.random.seed(42)

# Load datasets
df_main = pd.read_csv('spam_ham_dataset.csv')
df_aug = pd.read_csv('ham_augmented.csv')
df_test = pd.read_csv('test_emails.csv')

# Map labels consistently
df_main['label_num'] = df_main['label'].map({'ham': 0, 'spam': 1})
df_aug['label_num'] = df_aug['label'].map({'ham': 0})
df_test['label_num'] = df_test['label'].map({'ham': 0, 'spam': 1})

# Combine training data
df_train = pd.concat([df_main[['text', 'label_num']], df_aug[['text', 'label_num']]], ignore_index=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Clean and vectorize
X_clean = df_train['text'].apply(clean_text)
y = df_train['label_num']

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9,
    stop_words='english',
    max_features=5000
)
X_vec = vectorizer.fit_transform(X_clean)

# Split for internal evaluation
X_train, X_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train and calibrate model
base_model = RandomForestClassifier(n_estimators=300, random_state=42)
base_model.fit(X_train, y_train)

calibrated_rf = CalibratedClassifierCV(estimator=base_model, cv=5)
calibrated_rf.fit(X_train, y_train)

# Evaluate model
y_pred = calibrated_rf.predict(X_val)
print("✅ Accuracy on internal validation:", accuracy_score(y_val, y_pred))
print("📋 Classification Report:\n", classification_report(y_val, y_pred))

# 💾 Save model, vectorizer
joblib.dump(calibrated_rf, 'email_spam_rf_calibrated.pkl')
joblib.dump(vectorizer, 'email_tfidf_vectorizer.pkl')

# 🧪 Final evaluation on test_emails.csv
df_test['text_clean'] = df_test['text'].apply(clean_text)
X_test_final = vectorizer.transform(df_test['text_clean'])
y_test_final = df_test['label_num']
y_test_pred = calibrated_rf.predict(X_test_final)

print("\n📦 Final Evaluation on test_emails.csv")
print("✅ Accuracy:", accuracy_score(y_test_final, y_test_pred))
print("📋 Report:\n", classification_report(y_test_final, y_test_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test_final, y_test_pred))
