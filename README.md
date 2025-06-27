# 📧 Email Spam Filtering System

A machine learning-based Email Spam Filtering System developed using Python and Scikit-learn. This project classifies incoming emails as **Spam** or **Ham (legitimate)** using a trained Random Forest classifier with probability calibration for better reliability.

---

## 📖 About the Project

The Email Spam Filtering System aims to automatically detect and filter out spam emails from legitimate ones using machine learning techniques. It uses a **Random Forest Classifier** trained on labeled email datasets and applies **TF-IDF vectorization** for text feature extraction.

The system can be integrated with a local email server setup using **Postfix**, **Dovecot**, and **Procmail** for real-time spam filtering on received emails.

---

## ✨ Features

- Classifies emails as **Spam** or **Ham**
- Uses **TF-IDF** vectorizer for text feature extraction
- Implements **Calibrated Random Forest Classifier** for better probability estimates
- Command-line based prediction from raw email text
- Can be integrated with Linux email services for automatic email sorting
- Outputs classification report including **accuracy, precision, recall, and F1-score**

---

## 🛠️ Technologies Used

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Procmail (for email filtering)
- Postfix & Dovecot (optional integration for local email server)

---


📈 Results

    Accuracy: 97.01%

    Precision (Spam): 0.96

    Recall (Spam): 0.97

Classification report available via final_model.py execution.
🌱 Future Enhancements

    Integration with web-based email clients

    Use of deep learning models (e.g., LSTM/Transformer-based classifiers)

    Improved dataset diversity

    Interactive dashboard for email classification visualization

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit a pull request.


