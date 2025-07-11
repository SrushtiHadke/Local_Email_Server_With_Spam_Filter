#!/usr/bin/env python3
import pandas as pd
import re
import string
import joblib  # Changed from numpy for pickle support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Data Loading (unchanged)
def load_data():
    df_main = pd.read_csv('spam_ham_dataset.csv')
    df_aug = pd.read_csv('ham_augmented.csv')
    df_main['label_num'] = df_main['label'].map({'ham': 0, 'spam': 1})
    df_aug['label_num'] = df_aug['label'].map({'ham': 0})
    return pd.concat([df_main[['text', 'label_num']], df_aug[['text', 'label_num']]])

# Text Cleaning (unchanged)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    # Load and prepare data
    df_train = load_data()
    df_train['text_clean'] = df_train['text'].apply(clean_text)

    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=3500, stop_words='english')
    X = vectorizer.fit_transform(df_train['text_clean'])
    y = df_train['label_num']

    # Balance and split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train model
    model = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # Save models as .pkl files
    joblib.dump(vectorizer, 'email_vectorizer.pkl')
    joblib.dump(model, 'email_model.pkl')

    # Save threshold separately
    with open('threshold.txt', 'w') as f:
        f.write(str(0.45))  # Your threshold value

    print("Models saved as .pkl files")
