# Phishing Detector


## Overview
Phishing Emails pose a threat to personal belongings. With increasing digital communication comes increased risk of receiving Phishing Emails.

This is the reason why I decided to create, train and evaluate a personal Phishing Detector that gives me an assessment on whether a received Mail could be classified as Safe of Phishing Email. To train and evaluate the Phishing Detector, different machine learning algorithms from the Scikit library were used.

## Features
Apart from actually training and evaluating the model, I wanted to integrate the feature that a user can add his suspicious E-mails and find out the likelihood of it being a Phishing Mail.

## Requirements
- Python 3.8 or higher
- scikit-learn (v1.0.0+) for machine learning algorithms
- pandas for data handling
- numpy for numerical operations
- re (regular expressions) for text preprocessing
- pickle for model serialization

All dependencies can be installed using pip:
- pip install scikit-learn pandas numpy


## Usage
The Phishing Detector can be used in the following way:
- Add suspicious E-Mail messages into  the Examples.csv file
- Run the predict_examples.py script and observe the predicted outcome


## Dataset
The dataset used for training the model comes fron the following website: https://www.kaggle.com/datasets/subhajournal/phishingemails?select=Phishing_Email.csv

I downloaded the Phishing_Email.csv file from this site.

The file (every mail used for training) is structured in the following way:
- Number (int, starting from 0)
- E-Mail content (string)
- True value (Phishing of Safe)

## Model
Like already mentioned, the scikit-learn library was used for training and evaluating the data. More explicitely TfidfVectorizer was used for feature extraction while LinearSVC was used for actually training and evaluation.

How both of them work:

- TfidfVectorizer: Extracts terms from documents and finds a balance between how frequently a term appears in a document and how rare the term is across all documents. In other words, it gives high weight to terms that appear frequently in a particular document but also appear in relatively few documents overall.

- LinearSVC: This machine learning algorithm is used for the classification task, in this case splitting the Mails into Phishing and Safe. It does this by:
    - Finding the optimal hyperplane that best separate the data into different classes
    - Maximizes the distance between the closest data points from each class to the decision boundary
    - Creates a straight-line decision boundary by using a linear kernel
    - Classifies data according on which side of the kernel the data point lies

## Results
For the evaluation, scikit-learn's accuracy score and the classifiaction report was used.

For the testing, 20% of the total E-Mails were used

- Accuracy: (True Positives + True Negatives) / Total Samples
For the classification report:
- Precision: Of all emails predicted as phishing, how many were actually phishing
- Recall: Of all actual phishing emails, how many did the model correctly identify
- F1-Score: The harmonic mean of precision and recall
- Support: The number of actual occurrences of each class in the dataset

Accuracy: 0.97

The resulting table looks like this with some really promising stats:

                precision    recall  f1-score   support

    Safe Email       0.94      0.99      0.97      1383
    Phishing Email   0.99      0.96      0.98      2191

    accuracy                             0.97      3574
    macro avg        0.97      0.98      0.97      3574
    weighted avg     0.97      0.97      0.97      3574

## Example Predictions
The predictions of some of the E-Mails that aren't very obvious examples of Phishing or Safe Mail were classified incorrectly. That's why i added different confidence levels. Like this it makes it easier for the user to assess whether to trust the prediction of the script.

The decision value tells you the absolute distance from the data point to the decision line. The larger this distance, the more confidently you can say that the decision is correct. Furthermore, if the value is negative, it's classified as Phishing E-Mail and if it's positive it's classified as Safe E-Mail.

The following confidence levels were chosen:

- High -> Absolute distance > 1.0
- Rather Low -> 0.5 < Absolute distance < 1.0
- Very Low. Could be either -> Absolute distance < 0.5

## Limitations
The accuracy of the predictions can only be very high if the data used to train the models. Although the model has very good stats with the data it was trained with, it can be that some E-Mails might not fall into a clear pattern that helps the model to classify it with high certainty. If it's sometimes even hard to tell for a human being if a Mail is safe of a phishing attemt, it's also very difficult for a machine.

## Future Improvements
What could be done to improve the overall accuracy not only when evaluating the data that was used to train the model but also the accuracy for the prediction of the user E-Mails would be to expand the training data (Phishing_Email.csv) with some other examples that maybe represent real world Emails a bit better. It's important to note that this project was used more to learn to use the scikit-learning library for machine learning algorithms as I'm really interested in the topic.

