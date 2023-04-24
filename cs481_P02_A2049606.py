import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import math


# Download stopwords and punkt from nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text, remove_stopwords=True):
    # Remove non-alphanumeric characters, lowercase, and tokenize
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    # skip stopword removal if user wants to
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    else:
        tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# argument parsing
parser = argparse.ArgumentParser(description='Sentiment analysis using a custom Naive Bayes classifier')
parser.add_argument('skip_stopword_removal', nargs='?', choices=['YES', 'NO'], default='NO', help='YES to skip stopword removal and stemming during text preprocessing, NO otherwise')
args = parser.parse_args()




# Load data
file_path = 'reviews.csv'
df = pd.read_csv(file_path, delimiter=';')

# Combine text columns
combined_text = df['content'] + ' ' + df['replyContent']
df['combined_text'] = combined_text.apply(lambda x: preprocess(str(x), remove_stopwords=args.skip_stopword_removal == 'NO'))

# generate sentiment labels
df['sentiment'] = df['score'].apply(lambda x: 'positive' if x > 3 else 'negative')

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['sentiment'], test_size=0.2, random_state=42)

class NaiveBayes:
    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}

    def fit(self, X_train, y_train):
        classes, class_counts = np.unique(y_train, return_counts=True)
        self.class_probs = dict(zip(classes, class_counts / len(y_train)))

        for c in classes:
            word_counts = {}
            class_data = X_train[y_train == c]
            for doc in class_data:
                for word in doc.split():
                    word_counts[word] = word_counts.get(word, 0) + 1

            total_words = sum(word_counts.values())
            self.word_probs[c] = {word: (count + 1) / (total_words + len(word_counts))
                                  for word, count in word_counts.items()}

    def predict(self, X_test, return_probs=False):
        predictions = []
        prob_predictions = []
        for doc in X_test:
            posteriors = {}
            for c, prior in self.class_probs.items():
                likelihood = 1
                for word in doc.split():
                    likelihood *= self.word_probs[c].get(word, 1 / (sum(self.word_probs[c].values()) + len(self.word_probs[c])))

                posteriors[c] = prior * likelihood

            # Normalize probabilities
            prob_sum = sum(posteriors.values())

            if prob_sum != 0:
                normalized_probs = {c: p / prob_sum for c, p in posteriors.items()}
            else:
                normalized_probs = {c: 0 for c in posteriors.keys()}

            predictions.append(max(posteriors, key=posteriors.get))
            prob_predictions.append(normalized_probs)

        if return_probs:
            return predictions, prob_predictions
        else:
            return predictions

    def predict_with_prob(self, X_test):
        predictions = []
        probabilities = []
        for doc in X_test:
            posteriors = {}
            for c, prior in self.class_probs.items():
                likelihood = 1
                for word in doc.split():
                    likelihood *= self.word_probs[c].get(word, 1 / (
                                sum(self.word_probs[c].values()) + len(self.word_probs[c])))

                posteriors[c] = prior * likelihood

            predictions.append(max(posteriors, key=posteriors.get))
            probabilities.append(posteriors)

        return predictions, probabilities


print("Yaish, Sakher, A20496906 solution:")
print(f"Ignored pre-processing step: {'Stop Word Removal' if args.skip_stopword_removal == 'YES' else 'None'}")

naive_bayes = NaiveBayes()
print("\nTraining classifier…")
naive_bayes.fit(X_train, y_train)
print("Testing classifier…")
y_pred = naive_bayes.predict(X_test)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = recall_score(y_test, y_pred, pos_label='positive', average='binary')
specificity = tn / (tn + fp)
precision = precision_score(y_test, y_pred, pos_label='positive', average='binary')
negative_predictive_value = tn / (tn + fn)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='positive', average='binary')

# Output metrics
print("\nTest results / metrics:")
print(f"Number of true positives: {tp}")
print(f"Number of true negatives: {tn}")
print(f"Number of false positives: {fp}")
print(f"Number of false negatives: {fn}")
print(f"Sensitivity (recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Negative predictive value: {negative_predictive_value:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F-score: {f1:.4f}")

def get_label_percentages(labels):
    label_counts = labels.value_counts()
    total_count = len(labels)
    percentages = (label_counts / total_count) * 100
    return percentages

total_percentages = get_label_percentages(df['sentiment'])
print(total_percentages)
test_percentages = get_label_percentages(y_test)
print(test_percentages)
train_percentages = get_label_percentages(y_train)
print(train_percentages)

def classify_sentence(classifier, sentence, remove_stopwords=True):
    processed_sentence = preprocess(sentence, remove_stopwords=remove_stopwords)
    _, probabilities = classifier.predict([processed_sentence], return_probs=True)

    return probabilities[0]


print("\nEnter your sentence:")
while True:
    sentence = input("\nSentence S: ")

    if not sentence:
        print("Please enter a non-empty sentence.")
        continue

    probabilities = classify_sentence(naive_bayes, sentence, remove_stopwords=args.skip_stopword_removal == 'NO')

    class_a_prob = probabilities['negative']
    class_b_prob = probabilities['positive']

    if class_a_prob > class_b_prob:
        class_label = 'negative'
    else:
        class_label = 'positive'

    print(f"\n{sentence} was classified as {class_label}.")
    print(f"P(negative | S) = {class_a_prob:.4f}")
    print(f"P(positive | S) = {class_b_prob:.4f}")

    user_input = input("\nDo you want to enter another sentence [Y/N]? ").lower()
    if user_input != 'y':
        break
