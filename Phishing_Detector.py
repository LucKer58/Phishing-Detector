import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
    X_train, X_test, Y_train, Y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.001, max_features=10000, ngram_range=(1, 2), sublinear_tf=True, use_idf=True, strip_accents='unicode')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = LinearSVC(class_weight='balanced', dual=False, max_iter=2000)
    classifier.fit(X_train_tfidf, Y_train)

    return classifier, vectorizer, X_test_tfidf, Y_test



def evaluate_model(classifier, X_test_tfidf, y_test):
    y_predicted = classifier.predict(X_test_tfidf)

    print(y_predicted)

    accuracy = accuracy_score(y_test, y_predicted)
    print(f'Accuracy: {accuracy}')

    print(classification_report(y_test, y_predicted, target_names=["Safe Email", "Phishing Email"]))



def predict_phishing(cleaned_text, classifier, vectorizer):
    text_vector = vectorizer.transform([cleaned_text])
    
    decision_value = classifier.decision_function(text_vector)[0]
    
    prediction = classifier.predict(text_vector)[0]
    
    print(f'Prediction for this email: {prediction}')
    print(f'Decision value: {decision_value:.4f}')
    if abs(decision_value) > 1.0:
        print(f'Confidence: High')
    elif 0.5 < abs(decision_value) < 1.0:
        print(f'Confidence: Rather Low')
    else: print('Confidence: Very Low. Could be either')


def main():

    with open(r'c:\Users\lucie\OneDrive\Desktop\Phishing Detector\Phishing_Email.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        emails = list([row for row in csv_reader if len(row[1]) < 7000])
    processed_text = []
    labels = []
    for email in emails:
        cleaned_text, label = preprocess(email)
        processed_text.append(cleaned_text)
        labels.append(label)
    classifier, vectorizer, X_test_fdidf, Y_test = train_model(processed_text, labels)
    
    evaluate_model(classifier, X_test_fdidf, Y_test)


    with open(r'c:\Users\lucie\OneDrive\Desktop\Phishing Detector\Examples.csv', 'r', encoding='utf-8') as test_file:
        csv_reader2 = csv.reader(test_file)
        header2 = next(csv_reader2)
        examples = list(csv_reader2)
        
        for i, example in enumerate(examples):

            email_text = example[1]
            email_label = example[2]
            
            preprocess(email_text)
            
            print(f"\n--- Example {i} ---")
            print(f"True label: {email_label}")
            print(f"Email preview: {email_text[:70]}...")
            predict_phishing(cleaned_text, classifier, vectorizer)


if __name__ == "__main__":
    main()