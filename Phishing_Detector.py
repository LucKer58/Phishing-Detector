import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
csv.field_size_limit(20000000)

def preprocess(txt):
    email_text = txt[1]
    email_label = txt[2]

    cleaned_text = email_text.lower()
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', ' URL ', cleaned_text)
    cleaned_text = re.sub(r'\S+@\S+', ' EMAIL ', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text, email_label


def train_model(texts, labels):
    # Split, vectorize, train
    X_train, X_test, Y_train, Y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.001, max_features=10000, ngram_range=(1, 2), sublinear_tf=True, use_idf=True, strip_accents='unicode')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = LinearSVC(class_weight='balanced', dual=False, max_iter=2000)
    classifier.fit(X_train_tfidf, Y_train)

    return classifier, vectorizer, X_test_tfidf, Y_test



# 6. Evaluation Function
def evaluate_model(model, vectorizer, X_test, y_test):
    # Calculate metrics
    pass

# 7. Prediction Function
def predict_phishing(email_text, model, vectorizer):
    # Make predictions on new emails
    pass

def main():
    with open(r'c:\Users\lucie\OneDrive\Desktop\Phishing Detector\Phishing_Email.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
    header = next(csv_reader)
    emails = [row for row in csv_reader if len(row[1]) < 7000]
    processed_text = []
    labels = []
    for email in emails:
        cleaned_text, label = preprocess(email)
        processed_text.append(cleaned_text)
        labels.append(label)

if __name__ == "__main__":
    main()