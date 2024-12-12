import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary  # The vocabulary (list of words)
        self.vocab_size = len(vocabulary)
        self.class_prob = {}  # Class probabilities P(C)
        self.word_prob = defaultdict(lambda: defaultdict(float))  # P(w|C)
        self.class_word_counts = defaultdict(int)  # Count of words in each class
        self.class_counts = defaultdict(int)  # Count of messages in each class

    def fit(self, X_train, y_train):
        """
        Train the Naive Bayes classifier using the training data
        X_train: List of feature vectors (binary vectors)
        y_train: List of class labels corresponding to each feature vector
        """
        # Count the number of messages in each class
        total_messages = len(y_train)
        for label in y_train:
            self.class_counts[label] += 1

        # Calculate class probabilities P(C)
        for label in self.class_counts:
            self.class_prob[label] = self.class_counts[label] / total_messages

        # Count the occurrences of each word for each class
        for i, features in enumerate(X_train):
            label = y_train[i]
            self.class_word_counts[label] += sum(features)
            for j, feature in enumerate(features):
                if feature == 1:
                    word = self.vocabulary[j]
                    self.word_prob[label][word] += 1

        # Apply Laplace smoothing to compute P(w|C)
        for label in self.word_prob:
            for word in self.vocabulary:
                # Laplace smoothing: Add 1 to every word count
                self.word_prob[label][word] = (self.word_prob[label][word] + 1) / (self.class_word_counts[label] + self.vocab_size)

    def predict(self, X_test):
        """
        Predict the class labels for the test data
        X_test: List of feature vectors (binary vectors)
        Returns a list of predicted labels
        """
        predictions = []
        for features in X_test:
            class_scores = {}
            for label in self.class_prob:
                score = math.log(self.class_prob[label])  # Start with the log of P(C)
                for j, feature in enumerate(features):
                    if feature == 1:
                        word = self.vocabulary[j]
                        score += math.log(self.word_prob[label].get(word, 1 / (self.class_word_counts[label] + self.vocab_size)))
                class_scores[label] = score
            # Predict the class with the highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions
