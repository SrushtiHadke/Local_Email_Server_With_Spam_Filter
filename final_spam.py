#!/usr/bin/env python3
import joblib
import re
import sys
import numpy as np
from email import message_from_string

# Load models
try:
    vectorizer = joblib.load('/home/chatgpt/email_vectorizer.pkl')
    model = joblib.load('/home/chatgpt/email_model.pkl')
    with open('/home/chatgpt/threshold.txt') as f:
        threshold = float(f.read())
except Exception as e:
    print(f"Error loading models: {e}", file=sys.stderr)
    sys.exit(1)

# Text cleaning 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Whitelist rules
SAFE_PHRASES = [
    "birthday wishes", "invoice payment",
    "meeting reminder", "your order"
]

def is_safe(text):
    return any(phrase in text.lower() for phrase in SAFE_PHRASES)

def extract_email_text(raw_email):
    msg = message_from_string(raw_email)
    subject = msg.get('Subject', '')
    body = ''

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    body += part.get_payload(decode=True).decode(errors='ignore')
                except:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors='ignore')
        except:
            pass

    return f"{subject} {body}"

if __name__ == "__main__":
    raw_email = sys.stdin.read()
    full_text = extract_email_text(raw_email)
    cleaned = clean_text(full_text)

    if is_safe(cleaned):
        print("Ham", file=sys.stderr)
        sys.exit(1)

    X = vectorizer.transform([cleaned])
    spam_prob = 1 / (1 + np.exp(-model.decision_function(X)[0]))  # Sigmoid

    if spam_prob >= threshold:
        print(f"Spam ({spam_prob:.4f})", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"Ham ({spam_prob:.4f})", file=sys.stderr)
        sys.exit(1)
