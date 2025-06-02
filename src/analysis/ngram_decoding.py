
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler
from tqdm.auto import tqdm

from src.analysis.state_space import LabeledStateTrajectory


class TrigramDataset(Dataset):
    def __init__(self, embeddings, frame_metadata, class_key, class_labels):
        self.embeddings = embeddings
        self.frame_metadata = frame_metadata
        self.class_key = class_key
        self.class_labels = class_labels
        self.class_to_idx = {label: i for i, label in enumerate(class_labels)}
        self.frame_metadata["class_idx"] = self.frame_metadata[class_key].map(self.class_to_idx)

    def __len__(self):
        return len(self.frame_metadata)
    
    def __getitem__(self, idx):
        metadata_rows = self.frame_metadata.iloc[idx]
        frame_idx = metadata_rows.name
        class_idx = metadata_rows.class_idx

        x = torch.tensor(self.embeddings[frame_idx]).squeeze()
        y = torch.tensor(class_idx).long()

        return x, y
    

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
    

class MyCriterion(nn.Module):
    """
    inverse class weighted cross-entropy loss
    """
    def __init__(self, class_counts, device):
        super(MyCriterion, self).__init__()
        self.class_counts = class_counts
        self.class_weights = torch.tensor(1.0 / class_counts).float().to(device)
        self.class_weights /= self.class_weights.sum()

    def forward(self, logits, labels):
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return loss


class NgramDecoder:
    def __init__(self, lst: LabeledStateTrajectory, device="cuda", copy_embeddings=True, seed=42):
        self.lst = lst
        self.device = device
        self.copy_embeddings = copy_embeddings
        self.seed = seed
        self._primed = False

    def prepare_dataset(self, frame_metadata, class_key):
        self._classifier_data = frame_metadata.copy()
        self.class_key = class_key

        if self.copy_embeddings:
            embeddings = torch.tensor(self.lst.embeddings).to(self.device)
        else:
            embeddings = self.lst.embeddings

        # only keep the top 95% of the data
        class_cumprob = self._classifier_data[class_key].value_counts(normalize=True).cumsum()
        to_drop = class_cumprob[class_cumprob > 0.95].index
        self._classifier_data = self._classifier_data[~self._classifier_data[class_key].isin(to_drop)]

        self.class_labels = sorted(self._classifier_data[class_key].unique())
        num_classes = len(self.class_labels)
        print(f"Number of classes: {num_classes}")

        self.train_idx, test_idx = train_test_split(
            range(len(self._classifier_data)),
            test_size=0.3,
            random_state=self.seed,
        )
        self.val_idx, self.test_idx = train_test_split(
            test_idx,
            test_size=0.97,
            random_state=self.seed,
        )

        self.train_dataset = TrigramDataset(
            embeddings,
            self._classifier_data.iloc[self.train_idx],
            class_key,
            self.class_labels
        )
        self.val_dataset = TrigramDataset(
            embeddings,
            self._classifier_data.iloc[self.val_idx],
            class_key,
            self.class_labels
        )
        self.test_dataset = TrigramDataset(
            embeddings,
            self._classifier_data.iloc[self.test_idx],
            class_key,
            self.class_labels
        )

        self.class_counts = self._classifier_data.iloc[self.train_idx][class_key].value_counts().loc[self.class_labels].values

    def prime(self):
        self.criterion = MyCriterion(self.class_counts, self.device)
        self.model = MyModel(
            input_dim=self.lst.embeddings.shape[-1],
            output_dim=len(self.class_counts)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.global_step = 0
        self._primed = True

    def train(self, train_batch_size=128):
        if not self._primed:
            self.prime()
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=2048, shuffle=False)

        val_every = 500
        last_val_loss = float("inf")

        early_stopping_patience = 3
        early_stopping_counter = 0

        stop = False

        while True:
            if stop:
                break

            for X, Y in train_dataloader:
                self.optimizer.zero_grad()
                self.model.train()

                outputs = self.model(X.to(self.device))
                loss = self.criterion(outputs, Y.to(self.device))
                loss.backward()

                self.optimizer.step()

                if self.global_step % val_every == 0:
                    print(f"Iteration {self.global_step}, Training Loss: {loss.item()}")
                    with torch.no_grad():
                        self.model.eval()
                        val_loss = 0
                        for X, Y in tqdm(val_dataloader, leave=False, desc="Validation"):
                            outputs = self.model(X.to(self.device))
                            loss = self.criterion(outputs, Y.to(self.device))
                            val_loss += loss.item()
                        val_loss /= len(val_dataloader)
                        print(f"Iteration {self.global_step}, Validation Loss: {val_loss}")

                        if val_loss < last_val_loss:
                            last_val_loss = val_loss
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= early_stopping_patience:
                                print("Early stopping")
                                stop = True
                                break

                self.global_step += 1

    def predict(self):
        if not self._primed:
            self.prime()

        test_dataloader = DataLoader(self.test_dataset, batch_size=2048, shuffle=False)

        with torch.no_grad():
            self.model.eval()
            predictions = []
            for X, Y in tqdm(test_dataloader, leave=False, desc="Testing"):
                outputs = self.model(X.to(self.device))
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        test_df = self._classifier_data.iloc[self.test_idx].copy()
        test_df["predicted_class_idx"] = predictions
        test_df["predicted_class"] = test_df["predicted_class_idx"].map(dict(enumerate(self.class_labels)))
        test_df["correct"] = test_df["predicted_class"] == test_df[self.class_key]

        return test_df