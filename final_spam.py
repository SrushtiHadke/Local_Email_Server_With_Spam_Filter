#!/usr/bin/env python3

import joblib
import string
import sys
import re
import warnings
from email import message_from_string
from sklearn.exceptions import InconsistentVersionWarning
import os

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load model and vectorizer
model = joblib.load('/home/chatgpt/email_spam_rf_calibrated.pkl')
vectorizer = joblib.load('/home/chatgpt/email_tfidf_vectorizer.pkl')

# Use default threshold
threshold = 0.5

#  Clean incoming email
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

#  Read raw email
raw_email = sys.stdin.read()
email_msg = message_from_string(raw_email)
subject = email_msg.get('Subject', '')
body = ''

if email_msg.is_multipart():
    for part in email_msg.walk():
        if part.get_content_type() == 'text/plain':
            try:
                body += part.get_payload(decode=True).decode(errors='ignore')
            except:
                pass
else:
    try:
        body = email_msg.get_payload(decode=True).decode(errors='ignore')
    except:
        body = ''

full_text = clean_text(subject + ' ' + body)
msg_vec = vectorizer.transform([full_text])

#  Predict
proba = model.predict_proba(msg_vec)
spam_prob = proba[0][1]
pred = model.predict(msg_vec)
label = "Spam" if pred[0] == 1 else "Ham"

#  Output
print(f"[DEBUG] Probabilities (Ham, Spam): {proba[0]}", file=sys.stderr)
print(f"{label} (spam probability: {spam_prob:.3f})", file=sys.stderr)

# Return appropriate exit code for procmail
sys.exit(0 if label == "Spam" else 1)
