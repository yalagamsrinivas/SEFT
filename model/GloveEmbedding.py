
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.init as init
import io, os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet, brown
from collections import Counter

class glove_embedding(nn.Module):
    def __init__(self, path, embedding_dim, vocab_list):
        super(glove_embedding, self).__init__()
        self.embedding_dim = embedding_dim
        # Create word to index mapping from the provided vocab_list
        # Extend the vocab_list by 4 unspecified additional slots
        vocab_list = vocab_list + ['<extra_1>', '<extra_2>', '<extra_3>', '<extra_4>']
        self.word_to_index = {word: i for i, word in enumerate(vocab_list)}
        # Load embeddings and create the embedding matrix
        embeddings_dict = self.load_embeddings(path)
        self.embedding_matrix = self.create_embedding_matrix(embeddings_dict, vocab_list)

    def load_embeddings(self, path):
        embeddings_dict = {}
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                # Unify unknown word representations to 'UNK'
                if word in ['<unk>', 'unk']:
                    word = 'UNK'
                if word in self.word_to_index:  # Only load embeddings for words in vocab_list
                    vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                    embeddings_dict[word] = vector
        return embeddings_dict

    def create_embedding_matrix(self, embeddings_dict, vocab_list):
        embedding_matrix = torch.zeros(len(vocab_list), self.embedding_dim)
        for word, idx in self.word_to_index.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
        return nn.Parameter(embedding_matrix, requires_grad=False)  # Make embedding_matrix a nn.Parameter

    def forward(self, input_indices):
        # Flatten input_indices to 1D for embedding lookup, then reshape back to original dimensions with embedding size appended
        original_shape = input_indices.shape
        input_indices_flat = input_indices.flatten()
        embeddings_flat = self.embedding_matrix[input_indices_flat]
        new_shape = original_shape + (self.embedding_dim,)
        embeddings = embeddings_flat.view(*new_shape)
        return embeddings

