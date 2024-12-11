# Sentiment Analysis with Naive Bayes

This project implements a **Sentiment Analysis** model using a **custom Naive Bayes classifier** to classify text data as either **positive** or **negative** based on review scores. The model preprocesses the text, removes stopwords, applies stemming, and uses the Naive Bayes algorithm to predict sentiment.

## 📝 What It Is

The **Naive Bayes Sentiment Analysis** model classifies reviews into two categories:
- **Positive** sentiment (score > 3)
- **Negative** sentiment (score <= 3)

The project uses text preprocessing techniques, including tokenization, stopword removal, and stemming, to prepare the data for the Naive Bayes classifier. The model is built from scratch without relying on pre-built machine learning libraries, making it a custom solution.

### 🛠 How It’s Made

#### **Text Preprocessing**
The preprocessing step involves:
1. **Cleaning** the text by removing non-alphanumeric characters.
2. **Lowercasing** the text to maintain consistency.
3. **Tokenizing** the text into individual words.
4. **Removing stopwords** (optional, controlled via argument) to focus on meaningful words.
5. **Stemming** the tokens using the Porter Stemmer to reduce words to their root form.

#### **Naive Bayes Classifier**
The **Naive Bayes classifier** works by calculating the likelihood of each word belonging to either the **positive** or **negative** class. During training:
1. **Class Probabilities**: The probability of each sentiment (positive or negative) is computed based on the frequency of sentiment labels.
2. **Word Probabilities**: For each class, the probability of each word is computed based on its frequency within that class.

#### **Training & Testing**
1. **Training**: The model is trained using the preprocessed text data. For each class, word counts are tallied and normalized to form probability distributions.
2. **Testing**: The model is tested on a held-out set of reviews to predict whether each review is positive or negative based on its text. The predicted class is determined by computing the posterior probabilities for each class and selecting the class with the highest probability.

#### **Evaluation**
The model’s performance is evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The ability of the model to avoid false positives.
- **Recall**: The ability of the model to capture all true positives.
- **F1-Score**: A harmonic mean of precision and recall.
- **Confusion Matrix**: A table that describes the performance of the classifier on the test set.

### **Key Features**
- **Custom Implementation**: The Naive Bayes model is implemented from scratch, making it a learning exercise in building machine learning models.
- **Preprocessing Control**: You can choose to skip stopword removal and stemming if desired.
- **Interactive Sentiment Classification**: Users can input their own sentence and get sentiment predictions with class probabilities.

### Example:
Given a sentence like:
"I love this product!"

The model would output:
"I love this product!" was classified as positive.
P(negative | S) = 0.0342
P(positive | S) = 0.9658

## 📊 Data

The dataset used consists of the following columns:
- **content**: The review text.
- **replyContent**: The response to the review (if applicable).
- **score**: The score given in the review (1-5).

The text data is combined from the `content` and `replyContent` columns and preprocessed (tokenized, stopword removal, and stemming) to form a feature called `combined_text`. The sentiment is derived from the `score`, with values greater than 3 classified as **positive** and values less than or equal to 3 as **negative**.

## 🤖 Technologies Used

- **Python 3.x**: The primary programming language used for implementation.
- **pandas**: For data manipulation and reading CSV files.
- **numpy**: For numerical operations and calculations.
- **scikit-learn**: For evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- **nltk**: For text preprocessing, including tokenization, stopword removal, and stemming.
