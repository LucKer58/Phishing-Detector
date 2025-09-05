import csv
import re
import pickle
import os
from Phishing_Detector import preprocess

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

def load_model():
    """Load the trained classifier and vectorizer"""
    try:
        with open('classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return classifier, vectorizer
    except FileNotFoundError:
        return None, None

def main():
    
    classifier, vectorizer = load_model()
    if not classifier or not vectorizer:
        return
    
    print("\nAnalyzing examples from Examples.csv:")
    with open(r'c:\Users\lucie\OneDrive\Desktop\Phishing Detector\Examples.csv', 'r', encoding='utf-8') as test_file:
        csv_reader = csv.reader(test_file)
        examples = list(csv_reader)

            
    for i, example in enumerate(examples):
        print(f"\nExample {i}:")
        
        email_text = example[1]
        
        mock_email = ["", email_text, "Unknown"]
        cleaned_text, _ = preprocess(mock_email)
        
        print(f"Email preview: {email_text[:70]}...")
        predict_phishing(cleaned_text, classifier, vectorizer)
            

if __name__ == "__main__":
    main()
