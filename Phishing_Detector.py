import csv
import re
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