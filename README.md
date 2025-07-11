📧 Local Email Spam Filtering System

A machine learning-based Local Email Spam Filtering System developed using Python and Scikit-learn.
This project classifies incoming emails as Spam or Ham (legitimate) using a trained SGDClassifier with logistic regression loss for reliable probability estimation, and integrates seamlessly with a local email server setup for real-time email filtering.
📖 About the Project

This project aims to automatically detect and filter out spam emails from legitimate ones using machine learning techniques.
It uses an SGD Classifier (Log Loss) trained on labeled email datasets and applies TF-IDF vectorization for extracting text-based features.

Additionally, the system includes:

    SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

    Custom threshold control for adjusting spam detection sensitivity.

    Whitelisting of safe phrases to bypass spam classification for trusted content.

    Integration capability with Postfix, Dovecot, and Mutt for local email delivery and sorting.

✨ Features

    Classifies incoming emails as Spam or Ham

    Uses TF-IDF vectorization with bigrams for text feature extraction

    Employs SGDClassifier with Log Loss for faster, scalable model training

    Handles class imbalance using SMOTE

    Implements custom probability thresholding for classification decisions

    Whitelist-based fast Ham detection for known safe phrases (e.g., "birthday wishes", "invoice payment")

    Command-line compatible email prediction via stdin

    Easily pluggable into Postfix email pipelines

    Outputs classification report with accuracy, precision, recall, and F1-score

🛠️ Technologies Used

    Python 3.x

    Scikit-learn

    Imbalanced-learn

    Pandas

    NumPy

    Joblib

    Postfix & Dovecot (for local email server integration)

    Mutt (for email reading)

    Procmail (optional, for rule-based email handling)

📈 Results

Validation Accuracy: 97.89%

Precision (Spam): 0.96

Recall (Spam): 0.97

Full classification report available on executing final_model.py.
📦 Dataset

    spam_ham_dataset.csv → Original labeled spam and ham emails

    ham_augmented.csv → Manually augmented ham emails for improved dataset balance

🚀 How to Use

1️⃣ Train the Model

```
python3 final_model.py
```

Generates:

    email_model.pkl

    email_vectorizer.pkl

    threshold.txt

2️⃣ Classify Emails

```
echo "raw email content" | python3 final_spam.py
```

Outputs prediction and probability score via stderr and exit status for system integration.

3️⃣ Integrate with Postfix

    Configure Postfix to pipe emails through final_spam.py

    Route emails based on exit status (0 = Spam, 1 = Ham)

📑 Whitelist Rules

Automatically marks emails as Ham if they contain any of these phrases:

    birthday wishes

    invoice payment

    meeting reminder

    your order

🌱 Future Enhancements

    Add HTML and rich-text email parsing

    Integrate explainable AI (SHAP, LIME) for transparency in predictions

    Deploy a Gradio/Flask dashboard for live spam filtering stats

    Extend dataset with diverse email samples and attachments handling

    Experiment with Transformer-based models (e.g., BERT) for text classification

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repository, open an issue, or submit a pull request.
