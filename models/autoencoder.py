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
        self.gru = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        """
        Returns
        -------
        output: torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
        hidden: torch.Tensor, final hidden state for each sequence
            Shape: (1, batch_size, hidden_size)
        """
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocabulary: Vocabulary):
        super(DecoderRNN, self).__init__()
        self.vocabulary = vocabulary
        output_size = len(vocabulary)

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor, max_length=10):
        batch_size = encoder_outputs.size(0)

        decoder_input = torch.empty(*target_tensor.shape, dtype=torch.long).fill_(self.vocabulary.sos_token_id)
        # teacher forcing
        decoder_input[:, 1:] = target_tensor[:, :-1]

        decoder_input = self.embedding(decoder_input)
        decoder_outputs, decoder_hidden = self.lstm(decoder_input, encoder_hidden)

        # for i in range(max_length):
        #     decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
        #     decoder_outputs.append(decoder_output)

        #     if target_tensor is not None:
        #         # Teacher forcing: Feed the target as the next input
        #         decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
        #     else:
        #         # Without teacher forcing: use its own predictions as the next input
        #         _, topi = decoder_output.topk(1)
        #         decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = self.out(decoder_outputs).log_softmax(dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop
    

class Autoencoder(nn.Module):
    def __init__(self, hidden_size, vocabulary: Vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.encoder = EncoderRNN(len(vocabulary), hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocabulary)

    def forward(self, input_tensor, target_tensor=None, max_length=10):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor, max_length)
        return decoder_outputs, decoder_hidden
    
    def forward_loss(self, input_tensor, target_tensor, max_length=10):
        decoder_outputs, decoder_hidden = self.forward(
            input_tensor, target_tensor=target_tensor, max_length=max_length)
        loss = F.nll_loss(decoder_outputs.view(-1, len(self.vocabulary)), target_tensor.view(-1))
        return loss, decoder_outputs
    
    def encode(self, input_tensor):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        return encoder_outputs, encoder_hidden
    
    def prepare_string(self, text: str, max_length=10):
        words = text.strip().split(" ")

        if len(words) > max_length - 2:
            words = words[:max_length - 2]
        input_ids = [self.vocabulary.sos_token_id] + \
            [self.vocabulary.token2index[token] for token in words] + \
            [self.vocabulary.eos_token_id]
        return torch.tensor(input_ids, dtype=torch.long)
    

class VariationalAutoencoder(Autoencoder):

    def __init__(self, hidden_size, vocabulary: Vocabulary, kl_weight=1e-4):
        super().__init__(hidden_size, vocabulary)

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

        self.kl_weight = kl_weight

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, input_tensor):
        encoder_outputs, encoder_hidden = super().encode(input_tensor)
        
        mu = self.fc_mu(encoder_hidden[0])
        logvar = self.fc_logvar(encoder_hidden[0])
        return encoder_outputs, mu, logvar
    
    def vae_loss(self, decoder_outputs, target_tensor, mu, logvar):
        # Reconstruction loss
        recon_loss = F.nll_loss(decoder_outputs.view(-1, len(self.vocabulary)), target_tensor.view(-1), reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "loss": recon_loss + self.kl_weight * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    def forward(self, input_tensor):
        encoder_outputs, mu, logvar = self.encode(input_tensor)
        z = self.reparameterize(mu, logvar)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, (z, z), input_tensor)
        return decoder_outputs, mu, logvar
    
    def forward_loss(self, input_tensor, target_tensor):
        decoder_outputs, mu, logvar = self.forward(input_tensor)
        losses = self.vae_loss(decoder_outputs, target_tensor, mu, logvar)
        return losses, decoder_outputs