# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(nn.Module):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, input_dim, word_embeddings, prefix_embeddings):
        super(NeuralSentimentClassifier, self).__init__()

        # Initialize embeddings and vector size
        self.word_embeddings = word_embeddings
        self.prefix_embeddings = prefix_embeddings
        self.embedding_dim = input_dim

        # Define sizes for layers
        hidden_dim = input_dim * 2
        output_dim = 2

        # Initialize layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation_function = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize log softmax layer
        self.log_softmax = nn.LogSoftmax(dim=0)

        # Initialize weights using Xavier Glorot initialization
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [input_dim]-sized tensor of input data
        :return: an [output_dim]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # Pass the input through the first linear layer
        hidden_representation = self.input_layer(x)

        # Apply the activation function
        activated_output = self.activation_function(hidden_representation)

        # Pass the activated output through the second linear layer to get logits
        logits = self.output_layer(activated_output)

        # Apply log softmax to the logits to get log probabilities
        log_probabilities = self.log_softmax(logits)

        return log_probabilities

    def predict(self, example_words: List[str], has_typos: bool) -> int:
        """
        Does the prediction by doing the forward pass and returning the max of the log probabilities
        """
        average_embedding = self.prepare_input(example_words, has_typos)
        log_probs = self.forward(average_embedding)
        return np.argmax(log_probs.detach().numpy())

    def prepare_input(self, words: List[str], use_prefix: bool) -> torch.Tensor:
        """
        Prepare the input to the neural network.

        Approach for bonus question:
        1) Check if the word has its embedding in the word embedding and use it if it exists.
        2) If not, drop each character of the word one by one and check for modified word embedding.
        3) If none match, use the prefix embedding (first 3 characters of the word).

        :param words: List of words to process
        :param use_prefix: Boolean indicating whether to use prefix embeddings
        :return: A Tensor representing the average vector for the input words
        """
        total_embedding = np.zeros(self.embedding_dim)
        total_words = len(words)

        for word in words:
            word_embedding = self.word_embeddings.get_embedding(word)

            if use_prefix:
                if np.all(word_embedding == 0):  # Check if embedding is zero
                    # Try dropping one character at a time
                    found_embedding = False
                    for char_index in range(len(word), 2, -1):  # Start from the end
                        modified_word = word[:char_index] + word[char_index + 1:]
                        word_embedding = self.word_embeddings.get_embedding(modified_word)

                        if not np.all(word_embedding == 0):
                            found_embedding = True
                            break

                    # If no valid embedding found, use the prefix embedding
                    if not found_embedding:
                        if not self.prefix_embeddings:
                            continue
                        prefix = word[:3]
                        word_embedding = self.prefix_embeddings.get_embedding(prefix)

            # Accumulate the word embedding
            total_embedding += word_embedding

        # Average the embeddings over the number of words
        average_embedding = total_embedding / total_words
        return torch.from_numpy(average_embedding).float()



def get_prefix_embeddings(file_path: str) -> WordEmbeddings:
    """
    Reads ASCII-formatted word embeddings from a file and creates a WordEmbeddings object.
    Includes an UNK embedding initialized as a zero vector and a PAD token.

    :param file_path: Path to the file containing embeddings
    :return: A WordEmbeddings object containing the word vectors
    """
    with open(file_path, 'r') as file:
        indexer = Indexer()
        embeddings_list = []
        occurrence_counts = []

        # Add special tokens for padding and unknown words
        indexer.add_and_get_index("PAD")  # Index 0 for PAD token
        indexer.add_and_get_index("UNK")  # Index 1 for UNK token

        for line in file:
            line = line.strip()
            if line:  # Only process non-empty lines
                first_space_index = line.find(' ')
                word = line[:first_space_index]
                prefix = word[:3]

                # Convert the string of numbers into a numpy array
                vector_values = line[first_space_index + 1:].split()
                vector = np.array([float(value) for value in vector_values])

                # Initialize PAD and UNK vectors
                if not embeddings_list:
                    embeddings_list.append(np.zeros(vector.shape[0]))  # PAD vector
                    embeddings_list.append(np.zeros(vector.shape[0]))  # UNK vector
                    occurrence_counts.append(1)
                    occurrence_counts.append(1)

                # Process the prefix
                prefix_index = indexer.index_of(prefix)
                if prefix_index == -1:
                    new_index = indexer.add_and_get_index(prefix)
                    embeddings_list.append(vector)
                    occurrence_counts.append(1)
                else:
                    embeddings_list[prefix_index] += vector
                    occurrence_counts[prefix_index] += 1

    # Average the vectors
    averaged_vectors = [embeddings_list[i] / occurrence_counts[i] for i in range(len(embeddings_list))]

    return WordEmbeddings(indexer, np.array(averaged_vectors))


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line arguments
    :param train_exs: Training examples
    :param dev_exs: Development examples for evaluation during training
    :param word_embeddings: Loaded word embeddings
    :param train_model_for_typo_setting: Boolean to decide training model for typo setting
    :return: A trained NeuralSentimentClassifier model
    """

    # Set random seed for reproducibility
    random.seed(10)

    # Define hyperparameters
    num_classes = 2
    num_epochs = 1
    batch_size = args.batch_size
    initial_learning_rate = 0.005

    # Load prefix embeddings if typo setting is used
    prefix_embeddings = get_prefix_embeddings(args.word_vecs_path) if args.use_typo_setting else None

    # Initialize the model and optimizer
    input_size = word_embeddings.vectors[0].shape[0]
    model = NeuralSentimentClassifier(input_size, word_embeddings, prefix_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # Training loop for specified epochs
    for epoch in range(num_epochs):
        ex_indices = list(range(len(train_exs)))
        random.shuffle(ex_indices)
        total_loss = 0.0

        # Single instance per batch
        if batch_size == 1:
            for idx in ex_indices:
                x = model.prepare_input(train_exs[idx].words, args.use_typo_setting)
                y = train_exs[idx].label
                y_onehot = torch.zeros(num_classes).scatter_(0, torch.tensor([y], dtype=torch.int64), 1)

                model.zero_grad()
                log_probs = model.forward(x)
                loss = -log_probs.dot(y_onehot)
                loss.backward()

                total_loss += loss.item()
                optimizer.step()
        # Multiple instances per batch
        else:
            for batch_start in range(0, len(ex_indices), batch_size):
                batch_indices = ex_indices[batch_start: batch_start + batch_size]
                batch_x, batch_y = [], []

                for idx in batch_indices:
                    x = model.prepare_input(train_exs[idx].words, args.use_typo_setting)
                    y = train_exs[idx].label
                    batch_x.append(x)
                    batch_y.append(y)

                batch_x = torch.stack(batch_x)
                batch_y = torch.tensor(batch_y)
                y_onehot = torch.zeros(len(batch_y), num_classes).scatter_(1, batch_y.unsqueeze(1), 1)

                model.zero_grad()
                log_probs = model.forward(batch_x)
                loss = -torch.sum(log_probs * y_onehot)
                loss.backward()

                total_loss += loss.item()
                optimizer.step()

        print(f"Total loss on epoch {epoch}: {total_loss:.6f}")

    return model