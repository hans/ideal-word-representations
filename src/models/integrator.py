from dataclasses import dataclass
import random

from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers.file_utils import ModelOutput


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        lengths = lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output
    

class ContrastiveEmbeddingObjective(nn.Module):
    def __init__(self, tau=0.1):
        super(ContrastiveEmbeddingObjective, self).__init__()
        self.tau = tau

    def forward(self, embeddings, pos_embeddings, neg_embeddings):
        pos_dist = F.cosine_similarity(embeddings, pos_embeddings, dim=1)
        neg_dist = F.cosine_similarity(embeddings, neg_embeddings, dim=1)

        pos_loss = -torch.log(torch.exp(pos_dist / self.tau)).mean()
        neg_loss = -torch.log(torch.exp(-neg_dist / self.tau)).mean()

        loss = pos_loss + neg_loss

        return loss

@dataclass
class ContrastiveEmbeddingModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


class ContrastiveEmbeddingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tau=0.1):
        super(ContrastiveEmbeddingModel, self).__init__()
        self.rnn = RNNModel(input_dim, hidden_dim, output_dim)
        self.tau = tau

    def forward(self, example, example_length, pos, pos_length, neg, neg_length,
                return_loss=True, return_embeddings=False):
        embeddings, pos_embeddings, neg_embeddings = self.compute_batch_embeddings(
            example, example_length, pos, pos_length, neg, neg_length)
        loss = ContrastiveEmbeddingObjective(tau=self.tau)(embeddings, pos_embeddings, neg_embeddings)

        if not return_embeddings:
            return ContrastiveEmbeddingModelOutput(loss=loss)
        else:
            return ContrastiveEmbeddingModelOutput(
                loss=loss,
                embeddings=embeddings,
            )

    def compute_embeddings(self, example, example_length):
        embeddings = self.rnn(example, example_length)
        # Gather final embedding of each sequence
        embeddings = torch.gather(embeddings, 1, (example_length - 1).reshape(-1, 1, 1).expand(-1, 1, embeddings.shape[-1])).squeeze(1)
        return embeddings
        
    def compute_batch_embeddings(self, example, example_length, pos, pos_length, neg, neg_length):
        return self.compute_embeddings(example, example_length), \
                self.compute_embeddings(pos, pos_length), \
                self.compute_embeddings(neg, neg_length)


def get_sequence(F, start_index, end_index, max_length):
    if end_index - start_index + 1 > max_length:
        start_index = end_index - max_length + 1
    sequence = F[start_index:end_index + 1]
    
    # Pad on right if necessary
    if len(sequence) < max_length:
        pad_size = max_length - len(sequence)
        padding = torch.zeros(pad_size, F.shape[1])
        sequence = torch.cat((sequence, padding), dim=0)
    
    return sequence


def prepare_dataset(F, Q, S, max_length):
    dataset = []
    assert F.shape[0] == Q.shape[0] == S.shape[0]
    n_F = F.size(0)

    lengths = torch.arange(n_F) - S
    lengths = torch.minimum(lengths, torch.tensor(max_length))
    # TODO this is just a hack
    lengths[lengths == 0] = 1
    lengths[S == -1] = -1

    non_null_frames = (Q != -1).nonzero(as_tuple=True)[0]
    for i in tqdm(non_null_frames):
        if lengths[i] == -1:
            continue

        pos_indices = (Q == Q[i]).nonzero(as_tuple=True)[0]
        neg_indices = ((Q != -1) & (Q != Q[i])).nonzero(as_tuple=True)[0]

        if len(pos_indices) > 1 and len(neg_indices) > 0:
            pos_indices = pos_indices[pos_indices != i]
            pos_idx = random.choice(pos_indices)
            neg_idx = random.choice(neg_indices)

            # TODO ideally we'd have multiple positive/negative examples
            # per example, especially in sparser Q cases.

            # Extract sequences
            example_seq = get_sequence(F, S[i], i, max_length)
            pos_seq = get_sequence(F, S[pos_idx], pos_idx, max_length)
            neg_seq = get_sequence(F, S[neg_idx], neg_idx, max_length)

            dataset.append((example_seq, lengths[i],
                            pos_seq, lengths[pos_idx],
                            neg_seq, lengths[neg_idx]))

    ret = Dataset.from_dict({
        "example": [x[0] for x in dataset],
        "example_length": [x[1] for x in dataset],
        "pos": [x[2] for x in dataset],
        "pos_length": [x[3] for x in dataset],
        "neg": [x[4] for x in dataset],
        "neg_length": [x[5] for x in dataset],
    }).with_format("torch")

    # Sanity checks
    assert (ret["example_length"] > 0).all()
    assert (ret["pos_length"] > 0).all()
    assert (ret["neg_length"] > 0).all()

    return ret


# def prepare_batches(F, Q, S, max_length, batch_size=32):
#     dataset = []
#     assert F.shape[0] == Q.shape[0] == S.shape[0]
#     n_F = F.size(0)

#     lengths = torch.arange(n_F) - S
#     lengths = torch.minimum(lengths, torch.tensor(max_length))
#     # TODO this is just a hack
#     lengths[lengths == 0] = 1

#     for i in range(n_F):
#         pos_indices = (Q == Q[i]).nonzero(as_tuple=True)[0]
#         neg_indices = (Q != Q[i]).nonzero(as_tuple=True)[0]

#         if len(pos_indices) > 1 and len(neg_indices) > 0:
#             pos_indices = pos_indices[pos_indices != i]
#             pos_idx = random.choice(pos_indices)
#             neg_idx = random.choice(neg_indices)

#             # Extract sequences
#             example_seq = get_sequence(F, S[i], i, max_length)
#             pos_seq = get_sequence(F, S[pos_idx], pos_idx, max_length)
#             neg_seq = get_sequence(F, S[neg_idx], neg_idx, max_length)

#             dataset.append((example_seq, lengths[i],
#                             pos_seq, lengths[pos_idx],
#                             neg_seq, lengths[neg_idx]))

#     return DataLoader(TensorDataset(
#         # example frames and lengths
#         torch.stack([x[0] for x in dataset]), 
#         torch.stack([x[1] for x in dataset]), 

#         # positive frames and lengths
#         torch.stack([x[2] for x in dataset]),
#         torch.stack([x[3] for x in dataset]),

#         # negative frames and lengths
#         torch.stack([x[4] for x in dataset]),
#         torch.stack([x[5] for x in dataset])),
#         batch_size=batch_size, shuffle=True)


def compute_batched_rnn_loss(model, data_loader):
    total_loss = 0
    total_batches = 0

    for batch in data_loader:
        loss = model(batch)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


# # Example usage
# n_F, d = 100, 4  # Example dimensions
# F = torch.randn(n_F, d) * 3  # Random frame features
# Q = torch.randint(0, 10, (n_F,))  # Random frame matches
# S = torch.maximum(torch.tensor(0), torch.arange(n_F) - torch.randint(1, 10, (n_F,)))  # Random span indices
# max_length = 20  # Maximum sequence length for RNN

# # rnn_model = RNNModel(input_dim=d, hidden_dim=256, output_dim=d)
# model = ContrastiveEmbeddingModel(input_dim=d, hidden_dim=256, output_dim=d, tau=0.1)
# # data_loader = prepare_batches(F, Q, S, max_length, batch_size=32)
# # loss = compute_batched_rnn_loss(model, data_loader)
# # print(loss)

# # Build a test batch
# sample_idx, pos_sample_idx, neg_sample_idx = 37, 23, 85
# sample_length, pos_sample_length, neg_sample_length = 10, 8, 12
# example_batch = get_sequence(F, S[sample_idx], sample_idx, max_length).unsqueeze(0)
# pos_sample_batch = get_sequence(F, S[pos_sample_idx], pos_sample_idx, max_length).unsqueeze(0)
# neg_sample_batch = get_sequence(F, S[neg_sample_idx], neg_sample_idx, max_length).unsqueeze(0)
# example_lengths = torch.tensor([sample_length])
# pos_sample_lengths = torch.tensor([pos_sample_length])
# neg_sample_lengths = torch.tensor([neg_sample_length])
# with torch.no_grad():
#     embeddings, pos_embeddings, neg_embeddings = model.compute_batch_embeddings(example_batch, example_lengths, pos_sample_batch, pos_sample_lengths, neg_sample_batch, neg_sample_lengths)
# # Manually compute embeddings
# def compute_embedding_single(model, x, length):
#     return model.rnn.fc(model.rnn.rnn(x[:, :length, :])[0][:, -1, :])
# with torch.no_grad():
#     embeddings_manual = compute_embedding_single(model, example_batch, example_lengths)
#     pos_embeddings_manual = compute_embedding_single(model, pos_sample_batch, pos_sample_lengths)
#     neg_embeddings_manual = compute_embedding_single(model, neg_sample_batch, neg_sample_lengths)

#     # Check that the embeddings are the same
#     print(embeddings)
#     print(embeddings_manual)
#     print("//")
#     print(pos_embeddings)
#     print(pos_embeddings_manual)
#     print("//")
#     print(neg_embeddings)
#     print(neg_embeddings_manual)
#     torch.testing.assert_close(embeddings, embeddings_manual)
#     torch.testing.assert_close(pos_embeddings, pos_embeddings_manual)
#     torch.testing.assert_close(neg_embeddings, neg_embeddings_manual)
