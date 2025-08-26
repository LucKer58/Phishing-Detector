import csv
csv.field_size_limit(20000000)

# Open the CSV file
with open(r'c:\Users\lucie\OneDrive\Desktop\Phishing Detector\Phishing_Email.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    # Read the header
    header = next(csv_reader)
    print("Headers:", header)
    
    # Process the rows
    count = 0
    phishing_count = 0
    safe_count = 0
    
    for row in csv_reader:
        email_text = row[1]
        email_type = row[2]

        if len(email_text) < 7000:

            if email_type == 'Phishing Email':
                phishing_count += 1
            elif email_type == 'Safe Email':
                safe_count += 1

            count += 1
    
    print(f"\nTotal emails: {count}")
    print(f"Phishing emails: {phishing_count}")
    print(f"Safe emails: {safe_count}")
