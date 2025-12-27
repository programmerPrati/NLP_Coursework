# models.py
import numpy as np

from sentiment_data import *
from utils import *
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords # to remove stopwords
#import nltk

from collections import Counter

maxindex_bigram = 15000

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_vector = Counter()

        
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                # Add the word to the indexer and get its index
                index = self.indexer.add_and_get_index(word)
            else:
                # Just get the index without adding if not in the indexer
                index = self.indexer.index_of(word)
                if index == -1:  # Word not in indexer, so ignore it
                    continue
            if index > 0:
                feature_vector[index] += 1  # Increment the count for this word's index
        return feature_vector


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer


    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Inputs the sentences and then returns the feature vector for them.

        """
        feature_vector = Counter()


        for i in range(len(sentence) - 1):
            bigram = sentence[i].lower() + "|" + sentence[i+1].lower()

            if add_to_indexer:
                # Add the word to the indexer and get its index
                index = self.indexer.add_and_get_index(bigram)
            else:
                # Just get the index without adding if not in the indexer
                index = self.indexer.index_of(bigram)
                if index == -1:  # Word not in indexer, so ignore it
                    continue
            if index > 0 and index < maxindex_bigram:
                feature_vector[index] += 1  # Increment the count for this word's index
        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopWords = set(stopwords.words("english"))


    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Inputs the sentences and then returns the feature vector for them.

        """
        feature_vector = Counter()

        sentence_without_stopWords = []
        for word in sentence:
            if word.lower() not in self.stopWords:
                sentence_without_stopWords.append(word)

        for i in range(len(sentence_without_stopWords) - 1):
            bigram = sentence_without_stopWords[i].lower() + "|" + sentence_without_stopWords[i+1].lower()

            if add_to_indexer:
                # Add the word to the indexer and get its index
                index = self.indexer.add_and_get_index(bigram)
            else:
                # Just get the index without adding if not in the indexer
                index = self.indexer.index_of(bigram)
                if index == -1:  # Word not in indexer, so ignore it
                    continue
            if index > 0: # and index < maxindex_bigram:
                feature_vector[index] += 1  # Increment the count for this word's index
        return feature_vector


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector, featurizer):
        self.weights = weight_vector
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        features = list()
        feature_vector = self.featurizer.extract_features(sentence, False)
        features.append(feature_vector)

        sparse_f = [list(x.items()) for x in features]
        max_length = len(self.weights) - 1
        dense_f = np.zeros((len(sparse_f), max_length + 1))
        for i, x in enumerate(sparse_f):
            for feature, value in x:
                dense_f[i, feature] = value
        dot_product = np.dot(dense_f, self.weights)
        output = sigmoid(dot_product)

        if output[0] >= 0.5:
            return 1
        else:
            return 0

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    features = list()
    true_label = []
    for sent in train_exs:
        feature_vector = feat_extractor.extract_features(sent.words, True)
        features.append(feature_vector)

        # Get all the true labels out of the train examples and append them
        true_label.append(sent.label)
    sparse_f = [list(x.items()) for x in features]

    max_length = max((index for feature in sparse_f for index, _ in feature), default=0)

    dense_f = np.zeros((len(sparse_f), max_length + 1))
    for i, x in enumerate(sparse_f):
        for index, count in x:
            dense_f[i, index] = count

    epochs = 400
    lr = 4
    weights = np.zeros(max_length + 1)


    for iteration in range(epochs):
        dot_product = np.dot(dense_f, weights)
        probabilities = sigmoid(dot_product)
        gradient = np.dot(dense_f.T, (probabilities - true_label)) / len(true_label)
        weights -= lr * gradient

    return LogisticRegressionClassifier(weights, feat_extractor)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL or LR to run the appropriate system")
    return model
