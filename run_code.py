import NaiveBayes as nb


def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = set(line.strip() for line in file.readlines())
    return stopwords


def load_data(file_path, stopwords, vocabulary):
    """
    Loads and preprocesses the data, converting messages to feature vectors.
    file_path: Path to the text file containing the data.
    stopwords: Set of words to be removed.
    vocabulary: List of vocabulary words (sorted alphabetically).
    Returns a list of feature vectors (binary) and the corresponding class labels.
    """
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]

    # Split words, remove stopwords, and create feature vectors
    feature_vectors = []
    for line in data:
        words = set(line.split())
        words = words - stopwords  # Remove stopwords
        feature_vector = [1 if word in words else 0 for word in vocabulary]
        feature_vectors.append(feature_vector)

    return feature_vectors


def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file.readlines()]
    return labels


def run_code():
    # Load stopwords
    stopwords = load_stopwords('stoplist.txt')

    # Load training data
    train_data_raw = [line.strip() for line in open('traindata.txt').readlines()]
    train_labels = load_labels('trainlabels.txt')

    # Form the vocabulary from the training data (set of all unique words)
    all_words = set()
    for message in train_data_raw:
        all_words.update(message.split())  # Split the message into words and add to the set
    vocabulary = sorted(list(all_words))  # Sort vocabulary alphabetically

    # Now preprocess the training data using the vocabulary
    train_data = load_data('traindata.txt', stopwords, vocabulary)

    # Preprocess the test data
    test_data = load_data('testdata.txt', stopwords, vocabulary)
    test_labels = load_labels('testlabels.txt')

    # Initialize and train the Naive Bayes classifier
    nb_classifier = nb.NaiveBayesClassifier(vocabulary)
    nb_classifier.fit(train_data, train_labels)

    # Make predictions on the training data and testing data
    train_predictions = nb_classifier.predict(train_data)
    test_predictions = nb_classifier.predict(test_data)

    # Evaluate the accuracy of the classifier
    def accuracy(predictions, labels):
        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(labels)

    train_accuracy = accuracy(train_predictions, train_labels)
    test_accuracy = accuracy(test_predictions, test_labels)

    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')

# Run the main function
if __name__ == "__main__":
    run_code()
