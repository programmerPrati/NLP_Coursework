# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Embedding and positional encoding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)

        # Stack of Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Embed the input tokens and apply positional encoding
        embedded = self.embedding(indices)
        embedded = self.positional_encoding(embedded)

        # Pass through the stacked Transformer layers
        attention_maps = []
        for layer in self.layers:
            embedded, attn_map = layer(embedded)
            attn_map = attn_map.squeeze() if attn_map.dim() > 3 else attn_map

            attention_maps.append(attn_map)

        # Select the [CLS] token's embedding for classification
        cls_token_embedding = embedded[:, 0, :]  # Shape: (batch_size, d_model)

        # Project to the number of classes and apply softmax for probabilities
        logits = self.output_layer(cls_token_embedding)  # Shape: (batch_size, num_classes)
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # Shape: (batch_size, num_classes)

        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()

        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = max([factor for factor in range(1, 9) if d_model % factor == 0])

        self.d_k = d_model // self.num_heads  # Dimension per head

        # Linear layers to produce Q, K, and V matrices
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection after concatenating heads
        self.output_proj = nn.Linear(d_model, d_model)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

        # Layer norm
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_vecs):
        batch_size = input_vecs.size(0)

        # Project input to queries, keys, and values
        Q = self.query_proj(input_vecs)
        K = self.key_proj(input_vecs)
        V = self.value_proj(input_vecs)

        # Split each into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        attn_output, attn_map = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        attn_output = self.combine_heads(attn_output, batch_size)

        # Apply output projection
        attn_output = self.output_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Add & Norm: Residual connection and layer normalization
        attn_output = self.layer_norm1(input_vecs + attn_output)

        # Feedforward network with residual connection
        ff_output = self.ff(attn_output)
        ff_output = self.dropout(ff_output)
        output = self.layer_norm2(attn_output + ff_output)

        return output, attn_map
        # raise Exception("Implement me")

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Compute the scaled dot-product attention.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, depth)

    def combine_heads(self, x, batch_size):
        """
        Combine the heads back to a single tensor.
        """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # Extract vocabulary size from training data
    vocab_size = max({idx for ex in train for idx in ex.input_indexed}) + 1

    # Model parameters
    model = Transformer(vocab_size, num_positions=20, d_model=32, d_internal=16, num_classes=3, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()
    model.train()

    for epoch in range(5):
        total_loss = 0.0
        random.seed(epoch)
        for ex in random.sample(train, len(train)):  # Shuffle and loop through examples
            optimizer.zero_grad()
            log_probs, _ = model(ex.input_tensor)  # Forward pass
            loss = loss_fcn(log_probs, ex.output_tensor)  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/5, Loss: {total_loss:.4f}")

    model.eval()
    return model

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    '''model = Transformer(...)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            loss = loss_fcn(...) # TODO: Run forward and compute loss
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()
            loss_this_epoch += loss.item()
    model.eval()
    return model'''


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
