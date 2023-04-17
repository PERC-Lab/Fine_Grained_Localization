"""Module to represent the datasets"""

import math
import random
from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from tqdm import tqdm

from nltk import word_tokenize

from gensim.models import FastText


def get_end_idx(num_samples, ratio):
    return math.floor(num_samples * ratio)


class AbstractDataset(Dataset):

    def __init__(self):
        self.samples = []

    def get_raw_sample(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class AbstractCodeDataset(AbstractDataset, ABC):
    """Abstract class to represent Code dataset"""

    def __init__(self, split_ratio=(0.8, 0.1, 0.1), random=False) -> None:
        super().__init__()
        self.split_ratio = split_ratio

    def _split_samples(self):
        """Find and store the end index of
        train, valid, and test sets"""

        train_end_idx = get_end_idx(self.split_ratio[0], len(self.samples))
        val_end_idx = get_end_idx((self.split_ratio[0] + self.split_ratio[1]),
                                  len(self.samples))

        self.split_indices = {
            'train': (0, train_end_idx),
            'valid': (train_end_idx, val_end_idx),
            'test': (val_end_idx, len(self.samples))
        }

    def get_set(self, split_name: str):
        """Returns a map-style Dataset object containing samples
        of split_name ('train', 'valid', or 'test').

        Parameters
        ----------
        split_name : str
            Name of set. 'train' | 'valid' | 'test'

        Returns
        -------
        _RawCodeIterableDataset
            A wrapper dataset object
        """

        start_idx, end_idx = self.split_indices[split_name]
        dataset_samples = self.samples[start_idx:end_idx]
        return _RawCodeIterableDataset(dataset_samples,
                                       self.vocab,
                                       self.length,
                                       self.__getitem__)

    @abstractmethod
    def _init_dataset(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class _RawCodeIterableDataset(AbstractDataset, Dataset):
    def __init__(self, samples, vocab, length, getter_fn):
        self.samples = samples
        self.vocab = vocab
        self.length = length
        self.getter_fn = getter_fn

    def __getitem__(self, index):
        return self.getter_fn(index)


class ASTSeqDataset2(AbstractCodeDataset):
    """This class returns the code sample as a tuple of `start_terminal_node`,
     `non_terminal_nodes`, and `end_terminal_nodes`."""

    def __init__(self,
                 filename,
                 split_ratio,
                 length,
                 labels,
                 randomize=False,
                 delimiter="###") -> None:
        super().__init__(split_ratio, randomize)
        self.filename = filename
        self.length = length
        self.labelset = labels
        self.randomize_samples = randomize
        self.vocab = None
        self.num_classes = len(set(labels.values()))
        self._init_dataset(delimiter)

    @staticmethod
    def tokenize_nodes(ast_path):
        """Tokenizes the terminal and the non-terminal nodes in the path"""
        tokens = []
        split_tokens = ast_path.split(' ')
        terminal_node_start = split_tokens[0]
        terminal_node_end = split_tokens[-1]
        non_terminal_nodes = split_tokens[1].split('|')

        tokens.append(terminal_node_start)
        tokens.extend(non_terminal_nodes)
        tokens.append(terminal_node_end)

        return terminal_node_start, non_terminal_nodes, terminal_node_end

    @staticmethod
    def process_each_path(ast_path, counter):
        """This method tokenizes the terminal nodes,
        based on a single space, and the non-terminal nodes,
        based on "|". It also then pads the sequence at the end"""
        tokenized_nodes = ASTSeqDataset2.tokenize_nodes(ast_path)

        non_terminal_nodes = tokenized_nodes[1]

        nodes = []
        nodes.append(tokenized_nodes[0])
        nodes.extend(non_terminal_nodes)
        nodes.append(tokenized_nodes[-1])

        # Padding the sequence at the end instead of in the beginning.
        # nodes.extend(['\0' for f in range(9 - len(non_terminal_nodes))])

        counter.update(nodes)

        # return nodes
        return tokenized_nodes[0], non_terminal_nodes, tokenized_nodes[-1]

    @staticmethod
    def add_padding(src_code):
        """If the src_code contain paths that
        are just [0] then it expands them to be a list
        of 11 tokens
        """

        for i, el in enumerate(src_code):
            if el == [0]:
                src_code[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        return src_code

    def _init_dataset(self, delimiter):
        counter = Counter()

        with open(self.filename, 'r') as f:
            data = f.readlines()

        status_bar = tqdm(range(len(data)),
                          desc='Samples',
                          ascii=' >>>>>>>>>=',
                          bar_format='{desc}: {percentage:3.0f}% [{bar}]'
                                     ' {n_fmt}/{total_fmt} {desc}',
                          ncols=65,
                          colour='green')

        for i, el in enumerate(data):
            line = el.split(delimiter)

            sample_id = line[0]
            ast_paths = line[1][:-1]  # Removing the additional space

            paths = ast_paths.split(',')

            if paths[0] == '':
                continue

            # Tokenize the all nodes in each path
            tokenized_paths = [ASTSeqDataset2.process_each_path(
                f, counter) for f in paths]

            # Add padding if number of AST paths is than `length`
            pad = '\0, \0, \0, \0, \0, \0, \0, \0, \0, \0, \0'
            if len(tokenized_paths) < self.length:
                padding = [[pad]
                           for f in range(self.length - len(tokenized_paths))]
                tokenized_paths.extend(padding)
            elif len(tokenized_paths) >= self.length:
                tokenized_paths = tokenized_paths[:self.length-1]
                tokenized_paths.extend([[pad]])

            label = line[-1].strip()

            # tokenized path is a tuple of terminal_start_node,
            # non_terminal_paths, and terminal_end_node
            self.samples.append((sample_id, tokenized_paths, label))

            status_bar.update(1)

        self.vocab = Vocab(counter)

        if self.randomize_samples:
            random.shuffle(self.samples)

        self._split_samples()

    def __getitem__(self, index):

        # src_code is a list of lists, each inner list is an ast path
        _, src_code, label = self.samples[index]

        start_terminals = []
        end_terminals = []

        # Unify the tokenized terminal and non-terminal nodes. 
        for i, each_ast_path in enumerate(src_code):
            start_terminals.append(each_ast_path[0])
            end_terminals.append(each_ast_path[-1])

            if type(each_ast_path) is tuple:
                unified_ast_path = []

                unified_ast_path.append(each_ast_path[0])
                unified_ast_path.extend(each_ast_path[1])  # non_terminal_nodes
                unified_ast_path.append(each_ast_path[-1])  # end_terminal_node

                # Pad with zeros
                unified_ast_path.extend(
                    [0 for f in range(9 - len(each_ast_path[1]))])

                src_code[i] = unified_ast_path

        src_code_vocab = [[self.vocab[each_tok] for each_tok in ast_path]
                          for ast_path in src_code]

        start_terminal_vocab = [self.vocab[each_terminal]
                                for each_terminal in start_terminals]
        end_terminal_vocab = [self.vocab[each_terminal]
                              for each_terminal in end_terminals]

        src_code_vocab_padded = ASTSeqDataset2.add_padding(src_code_vocab)

        src_code_tensor = torch.tensor(
            src_code_vocab_padded, dtype=torch.int64)
        start_terminal_tensor = torch.tensor(
            start_terminal_vocab, dtype=torch.int64)
        end_terminal_tensor = torch.tensor(
            end_terminal_vocab, dtype=torch.int64)

        label = self.labelset[label]

        return (start_terminal_tensor,
                src_code_tensor,
                end_terminal_tensor,
                label)
