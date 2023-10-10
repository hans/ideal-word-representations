from typing import TypeAlias, Optional

from jaxtyping import Float
import torch
from torch import nn


T: TypeAlias = torch.Tensor

class NCERankerModel(nn.Module):
    def __init__(self, embedding_size, context_size, hidden_size):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(embedding_size + context_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, 1)
        
    def encode(self, embedding, context):
        return self.hidden(torch.cat([embedding, context], dim=-1))
        # return torch.cat([embedding, context], dim=-1)

    def forward(self, embedding, context):
        return self.out(self.encode(embedding, context))
    

class NCELoss(nn.Module):
    def __init__(self, model: NCERankerModel, embedding_model,
                 negative_samples: int,
                 vocab_mask: Optional[T] = None):
        """
        `vocab_mask` is an optional mask to specify which tokens are allowed as negative samples
        """
        super().__init__()
        self.model = model
        self.embedding_model = embedding_model
        self.negative_samples = negative_samples
        self.vocab_mask = vocab_mask
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, embedding, gt_token, next_token_dist: Float[T, "batch vocab_size"]):
        # TODO ensure that gt token is not a member of negative samples
        batch_size = embedding.shape[0]

        # next_token_dist: batch * vocab_size
        # Sample negative samples from the next-token distribution
        if self.vocab_mask is not None:
            next_token_dist[:, ~self.vocab_mask] = -float("inf")
        next_token_dist -= torch.max(next_token_dist, dim=-1, keepdim=True)[0]
        next_token_dist = torch.exp(next_token_dist)
        negative_samples = torch.multinomial(next_token_dist, self.negative_samples)

        logits_positive = self.model(embedding, self.embedding_model(gt_token))

        negative_embeddings = self.embedding_model(negative_samples)
        embedding_expanded = embedding.unsqueeze(1) \
            .expand(batch_size, self.negative_samples, embedding.shape[-1])
        logits_negative = self.model(embedding_expanded, negative_embeddings)

        all_logits = torch.cat([logits_positive, logits_negative.reshape(batch_size * self.negative_samples, -1)])
        all_labels = torch.cat([torch.ones_like(logits_positive), torch.zeros_like(logits_negative).reshape(-1, 1)])
        loss = self.bce(all_logits, all_labels)
        # TODO double check logic
        # logit_true = logits_positive.unsqueeze(1).expand_as(logits_negative) \
        #     - logits_negative
        # logit_true = logits_positive
        # logit_true = logit_true.reshape(-1)
        # label = torch.ones_like(logit_true)
        # loss = self.bce(logit_true, label)
        return loss