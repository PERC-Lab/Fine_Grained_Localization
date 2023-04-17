"""Module with all the models"""

from typing import Dict

import torch
from torch import nn
from torchtext import vocab
from torch.nn import functional as F


class AST2Class(nn.Module):
    """This is a baseline model for predicting a class"""

    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_hidden,
                 n_output,
                 n_layers,
                 batch_size,
                 drop_p=0.8):

        super().__init__()

        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.fc_1 = nn.Linear(3 * embed_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            n_hidden,
                            n_layers,
                            batch_first=True,
                            dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc_2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=1)

    def load_embedding(self, vocab: vocab, embedding_dict: Dict):
        """Creates embedding layer from dataset vocab and em

        Parameters
        ----------
        vocab : vocab
            Vocabulary of the dataset (not from one of the sets)
        embedding_dict : Dict
            FastText's word2vector model where key is token and value
            is the embedding weight in a numpy array
        """
        vocab_size = len(vocab)
        embedding_dim = embedding_dict.vector_size

        embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        for i, el in enumerate(vocab.itos):
            embedding_matrix[i] = torch.from_numpy(embedding_dict[el].copy())

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def forward(self, input_words):
        num_paths = input_words.shape[1]  # Only accepts batch inputs

        embedded_words = self.embedding(input_words)
        embedded_words_reshaped = embedded_words.view(-1,
                                                      num_paths,
                                                      3 * self.embed_dim)
        fc_out_1 = self.fc_1(embedded_words_reshaped)
        lstm_out, h = self.lstm(fc_out_1)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1,
                                              self.n_hidden)
        fc_out = self.fc_2(lstm_out)
        softmax_out = self.softmax(fc_out)
        softmax_reshaped = softmax_out.view(self.batch_size,
                                            -1,
                                            self.n_output)
        softmax_last = softmax_reshaped[:, -1]

        return softmax_last, h

    def init_hidden(self):
        device = "cuda"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device))

        return h


class AST2Class2(nn.Module):
    """This follows the Code2Seq style of forward for classification"""

    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_hidden,
                 n_output,
                 n_layers,
                 batch_size,
                 drop_p=0.8):

        super().__init__()

        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # self.fc_1 = nn.Linear(11 * embed_dim, embed_dim)
        self.lstm = nn.LSTM(1100,
                            100,
                            n_layers,
                            batch_first=True,
                            dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc_2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def load_embedding(self, vocab: vocab, embedding_dict: Dict):
        """Creates embedding layer from dataset vocab and em

        Parameters
        ----------
        vocab : vocab
            Vocabulary of the dataset (not from one of the sets)
        embedding_dict : Dict
            FastText's word2vector model where key is token and value
            is the embedding weight in a numpy array
        """
        vocab_size = len(vocab)
        embedding_dim = embedding_dict.vector_size

        embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        for i, el in enumerate(vocab.itos):
            embedding_matrix[i] = torch.from_numpy(embedding_dict[el].copy())

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def forward(self, start_terminals, input_words, end_terminals):
        """input_words.shape = (batch_size, num_paths, num_tokens)
        """
        num_paths = input_words.shape[1]  # Only accepts batch inputs

        embedded_words = self.embedding(input_words)
        # (batch_size, num_paths, num_tokens, embed_size)

        start_terminals_embedded = self.embedding(start_terminals)
        end_terminals_embedded = self.embedding(end_terminals)

        embedded_words_reshaped = embedded_words.view(-1,
                                                      num_paths,
                                                      11 * self.embed_dim)
        # (batch_size, num_paths, num_tokens * embed_size)

        lstm_out, h = self.lstm(embedded_words_reshaped)
        # lstm_out (batch_size, num_paths, hidden_size),
        # h (num_layers, batch_size, hidden_size)

        lstm_out = self.dropout(lstm_out)
        # (batch_size, num_paths, hidden_size)

        concatenated_context = torch.cat(
            [lstm_out, start_terminals_embedded, end_terminals_embedded])
        reshaped_concatenated_context = concatenated_context.view(
            -1, self.n_hidden)

        fc_out_context = self.fc_2(reshaped_concatenated_context)

        sigmoid_out_context = self.sigmoid(fc_out_context)

        sigmoid_out_context_reshaped = sigmoid_out_context.view(
            self.batch_size, -1)

        sigmoid_last = sigmoid_out_context_reshaped[:, -1]

        return sigmoid_last, h

    def init_hidden(self):
        device = "cuda"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device))

        return h


class AST2Class2WithAttention(nn.Module):

    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_hidden,
                 n_output,
                 n_layers,
                 batch_size,
                 drop_p=0.8):

        super().__init__()

        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.fc_1 = nn.Linear(11 * embed_dim, embed_dim)
        self.lstm = nn.LSTM(1100,
                            100,
                            n_layers,
                            batch_first=True,
                            dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc_2 = nn.Linear(n_hidden, n_hidden)
        self.fc_3 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def load_embedding(self, vocab: vocab, embedding_dict: Dict):
        """Creates embedding layer from dataset vocab and em

        Parameters
        ----------
        vocab : vocab
            Vocabulary of the dataset (not from one of the sets)
        embedding_dict : Dict
            FastText's word2vector model where key is token and value
            is the embedding weight in a numpy array
        """
        vocab_size = len(vocab)
        embedding_dim = embedding_dict.vector_size

        embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        for i, el in enumerate(vocab.itos):
            embedding_matrix[i] = torch.from_numpy(embedding_dict[el].copy())

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def attention(self, h, lstm_out):
        hidden = h.squeeze(0)
        # (batch_size, 2, hidden_size)
        # 2 is for hidden_vector and context_vector

        hidden_changed = hidden.permute(1, 2, 0)
        # (batch_size, hidden_size, 2)

        mat_mul_result = torch.bmm(lstm_out, hidden_changed)
        # (batch_size, hidden_size, 2)

        attention_weights = mat_mul_result.squeeze(2)
        # (batch_size, hidden_size, 2)

        attention_weights = attention_weights[:, :, 0]
        # (batch_size, hidden_size). First layer implies the hidden vector

        soft_attention_weights = F.softmax(attention_weights, dim=1)
        # (batch_size, hidden_size)

        unsqueeze_soft_attention = soft_attention_weights.unsqueeze(2)
        # (batch_size, hidden_size, 1)

        transposed_lstm_output = lstm_out.transpose(1, 2)
        # (batch_size, hidden_size, num_paths)
        # TODO: Check to see if I should not transpose the above matrix?!!

        mat_mul_second_result = torch.bmm(
            transposed_lstm_output, unsqueeze_soft_attention)
        # (batch_size, hidden_size, 1)

        new_hidden_state = mat_mul_second_result.squeeze(2)
        # (batch_size, hidden_size)

        return new_hidden_state

    def forward(self, start_terminals, input_words, end_terminals):
        """input_words.shape = (batch_size, num_paths, num_tokens)
        """

        num_paths = input_words.shape[1]  # (batch_size, num_paths, num_tokens)

        embedded_words = self.embedding(input_words)
        # (batch_size, num_paths, num_tokens, embedding_dim)

        start_terminals_embedded = self.embedding(start_terminals)
        # (batch_size, num_paths, embedding_dim)

        end_terminals_embedded = self.embedding(end_terminals)
        # (batch_size, num_paths, embedding_dim)

        embedded_words_reshaped = embedded_words.view(-1,
                                                      num_paths,
                                                      11 * self.embed_dim)
        # (batch_size, num_paths, embedding_dim * num_tokens)

        lstm_out, (h, c) = self.lstm(embedded_words_reshaped)
        # lstm_out (batch_size, num_paths, hidden_size)
        # h (batch_size, hidden_size)

        lstm_out = self.dropout(lstm_out)
        concatenated_context = torch.cat(
            [lstm_out, start_terminals_embedded, end_terminals_embedded])
        # (batch_size * 3, num_paths, embedding_dim)

        reshaped_concatenated_context = concatenated_context.view(
            -1, self.n_hidden)
        # (batch_size * 3 * num_paths, hidden_size)
        # TODO: Use reshaped_concatenated_context in the attention

        attention_weights = self.attention(h, lstm_out)

        # # Attention
        # hidden = h.squeeze(0)
        # # (batch_size, 2, hidden_size)
        # # 2 is for hidden_vector and context_vector

        # hidden_changed = hidden.permute(1, 2, 0)
        # # (batch_size, hidden_size, 2)

        # mat_mul_result = torch.bmm(lstm_out, hidden_changed)
        # print(f"mat_mul_result: {mat_mul_result.shape}")
        # # (batch_size, hidden_size, 2)

        # attention_weights = mat_mul_result.squeeze(2)
        # print(f"attention_weights: {attention_weights.shape}")
        # # (batch_size, hidden_size, 2)

        # attention_weights = attention_weights[:, :, 0]
        # print(f"attention_weights_first_layer: {attention_weights.shape}")
        # # (batch_size, hidden_size). First layer implies the hidden vector

        # soft_attention_weights = F.softmax(attention_weights)
        # # (batch_size, hidden_size)

        # unsqueeze_soft_attention = soft_attention_weights.unsqueeze(2)
        # # (batch_size, hidden_size, 1)

        # transposed_lstm_output = lstm_out.transpose(1, 2)
        # # (batch_size, hidden_size, num_paths)
        # # TODO: Check to see if I should not transpose the above matrix?!!

        # mat_mul_second_result = torch.bmm(
        #     transposed_lstm_output, unsqueeze_soft_attention)
        # # (batch_size, hidden_size, 1)

        # new_hidden_state = mat_mul_second_result.squeeze(2)
        # # (batch_size, hidden_size)

        # TODO: Add a non-linearity
        fc_out = self.fc_2(attention_weights)
        # (batch_size, num_fc2)

        # TODO: Add a non-linearity
        fc_last_out = self.fc_3(fc_out)
        # (batch_size, 1)

        fc_out_reshaped = fc_last_out.squeeze(1)
        # (batch_size)

        sigmoid_out = self.sigmoid(fc_out_reshaped)
        # (batch_size)

        return sigmoid_out, h

    def init_hidden(self):
        device = "cuda"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device))

        return h


class BiLSTM2ClassWithAttention(nn.Module):

    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_hidden,
                 n_output,
                 n_layers,
                 batch_size,
                 drop_p=0.8):

        super().__init__()

        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.fc_1 = nn.Linear(11 * embed_dim, embed_dim)
        self.lstm = nn.LSTM(1100,
                            100,
                            n_layers,
                            batch_first=True,
                            dropout=drop_p,
                            bidirectional=True)
        self.dropout = nn.Dropout(drop_p)
        self.fc_2 = nn.Linear(n_hidden * 2, n_hidden) # 2 for bi-directional LSTM
        self.fc_3 = nn.Linear(n_hidden, n_output)
        self.attention_layer = nn.Linear(2 * n_hidden, n_hidden) # 2 for bi-directional LSTM
        self.concat_linear = nn.Linear(3 * n_hidden, n_hidden) # 3 to make it fit
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def load_embedding(self, vocab: vocab, embedding_dict: Dict):
        """Creates embedding layer from dataset vocab and em

        Parameters
        ----------
        vocab : vocab
            Vocabulary of the dataset (not from one of the sets)
        embedding_dict : Dict
            FastText's word2vector model where key is token and value
            is the embedding weight in a numpy array
        """
        vocab_size = len(vocab)
        embedding_dim = embedding_dict.vector_size

        embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        for i, el in enumerate(vocab.itos):
            embedding_matrix[i] = torch.from_numpy(embedding_dict[el].copy())

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def attention(self, h, lstm_out):

        # print(f"**Inside Attention**")

        # print(f"h:{h.shape}")
        # print(f"lstm_out:{lstm_out.shape}")

        hidden = h.view(self.n_layers, 2, self.batch_size, self.n_hidden)
        # hidden (num_layers, num_directions, bs, hidden_dim)
        # print(f"hidden: {hidden.shape}")

        hidden_1, hidden_2 = hidden[0], hidden[1]
        # print(f"hidden_1: {hidden_1.shape}")

        # print(f"hidden_2: {hidden_2.shape}")
        
        # concatenate the two hidden states and sum them along the num_directions axis
        final_hidden_state = torch.cat((hidden[0], hidden[1]), 0)
        # final_hidden_state (num_layers, bs, hidden_dim)
        # print(f"final_hidden_state: {final_hidden_state.shape}")

        final_hidden_state_summed = final_hidden_state.sum(dim=0)
        # final_hidden_state_summed (bs, hidden_dim)
        # print(f"final_hidden_state_summed: {final_hidden_state_summed.shape}")

        final_hidden_state_unsqueezed = final_hidden_state_summed.unsqueeze(-1)
        # print(f"final_hidden_state_unsqueezed: {final_hidden_state_unsqueezed.shape}")

        attention_output = self.attention_layer(lstm_out)
        # print(f"attention_output: {attention_output.shape}")
        # (batch_size, seq_len, hidden_dim)

        mat_mul_result = torch.bmm(attention_output, final_hidden_state_unsqueezed)
        # print(f"mat_mul_result:{mat_mul_result.shape}")
        # (batch_size, hidden_size, 1)

        attention_weights = mat_mul_result.squeeze(2)
        # (batch_size, hidden_size, )
        # print(f"attention_weights: {attention_weights.shape}")

        soft_attention_weights = F.softmax(attention_weights, dim=1)
        # (batch_size, hidden_size)

        unsqueeze_soft_attention = soft_attention_weights.unsqueeze(2)
        # (batch_size, hidden_size, 1)

        transposed_lstm_output = lstm_out.transpose(1, 2)
        # (batch_size, hidden_size, num_paths)
        # TODO: Check to see if I should not transpose the above matrix?!!

        mat_mul_second_result = torch.bmm(
            transposed_lstm_output, unsqueeze_soft_attention)
        # (batch_size, hidden_size, 1)
        # print(f"mat_mul_second_result: {mat_mul_second_result.shape}")

        new_hidden_state = mat_mul_second_result.squeeze(2)
        # (batch_size, hidden_size * num_directions) This is context on the github website
        # print(f"new_hidden_state: {new_hidden_state.shape}")

        concatenated_hidden_states = torch.cat((new_hidden_state, final_hidden_state_summed ), dim=1)
        # print(f"concatenated_hidden_states: {concatenated_hidden_states.shape}")

        concat_linear_output = torch.tanh(self.concat_linear(concatenated_hidden_states))
        # print(f"concat_linear_output: {concat_linear_output.shape}")


        return new_hidden_state

    def forward(self, start_terminals, input_words, end_terminals):
        """input_words.shape = (batch_size, num_paths, num_tokens)
        """

        num_paths = input_words.shape[1]  # (batch_size, num_paths, num_tokens)
        # print(f"input_words: {input_words.shape}")

        # print(f"num_paths: {num_paths}")

        embedded_words = self.embedding(input_words)
        # print(f"embedded_words: {embedded_words.shape}")
        # (batch_size, num_paths, num_tokens, embedding_dim)

        start_terminals_embedded = self.embedding(start_terminals)
        # (batch_size, num_paths, embedding_dim)

        end_terminals_embedded = self.embedding(end_terminals)
        # (batch_size, num_paths, embedding_dim)

        embedded_words_reshaped = embedded_words.view(-1,
                                                      num_paths,
                                                      11 * self.embed_dim)
        # (batch_size, num_paths, embedding_dim * num_tokens)

        lstm_out, (h, c) = self.lstm(embedded_words_reshaped)
        # print(f"lstm_out: {lstm_out.shape}")
        # print(f"h: {h.shape}")
        # lstm_out (batch_size, num_paths, hidden_size)
        # h (batch_size, hidden_size)

        lstm_out = self.dropout(lstm_out)
        # concatenated_context = torch.cat(
        #     [lstm_out, start_terminals_embedded, end_terminals_embedded])
        # (batch_size * 3, num_paths, embedding_dim)

        # reshaped_concatenated_context = concatenated_context.view(
        #     -1, self.n_hidden)
        # (batch_size * 3 * num_paths, hidden_size)
        # TODO: Use reshaped_concatenated_context in the attention

        attention_weights = self.attention(h, lstm_out)

        fc_out = self.fc_2(attention_weights)
        # (batch_size, num_fc2)

        fc_last_out = self.fc_3(fc_out)
        # (batch_size, 1)

        fc_out_reshaped = fc_last_out.squeeze(1)
        # (batch_size)

        sigmoid_out = self.sigmoid(fc_out_reshaped)
        # (batch_size)

        return sigmoid_out, h

    def init_hidden(self):
        device = "cuda"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device))

        return h
