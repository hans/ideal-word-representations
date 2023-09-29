"""
Defines sequence-to-sequence autoencoder model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Vocabulary


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocabulary: Vocabulary):
        super(DecoderRNN, self).__init__()
        self.vocabulary = vocabulary
        output_size = len(vocabulary)

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, max_length=10):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.vocabulary.sos_token_id)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    

class Autoencoder(nn.Module):
    def __init__(self, hidden_size, vocabulary: Vocabulary):
        super(Autoencoder, self).__init__()
        self.encoder = EncoderRNN(len(vocabulary), hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocabulary)

    def forward(self, input_tensor, target_tensor=None, max_length=10):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor, max_length)
        return decoder_outputs, decoder_hidden