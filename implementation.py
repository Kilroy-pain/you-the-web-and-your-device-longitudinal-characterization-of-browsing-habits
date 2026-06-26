import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class ClickstreamDataset(Dataset):
    def __init__(self, clickstreams):
        """
        Initialize the dataset with clickstream data.
        :param clickstreams: List of clickstreams, where each clickstream is a list of page IDs.
        """
        self.clickstreams = clickstreams

    def __len__(self):
        return len(self.clickstreams)

    def __getitem__(self, idx):
        """
        Return a single clickstream and its next page prediction target.
        """
        clickstream = self.clickstreams[idx]
        input_sequence = clickstream[:-1]
        target = clickstream[1:]
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class ClickstreamModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        A simple RNN-based model for predicting the next page in a clickstream.
        :param vocab_size: Number of unique pages (vocabulary size).
        :param embedding_dim: Dimension of the embedding layer.
        :param hidden_dim: Dimension of the RNN hidden state.
        """
        super(ClickstreamModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, sequence_length).
        :return: Output logits of shape (batch_size, sequence_length, vocab_size).
        """
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        logits = self.fc(rnn_out)
        return logits

def generate_dummy_clickstreams(num_users, max_length, vocab_size):
    """
    Generate dummy clickstream data for testing.
    :param num_users: Number of users (clickstreams).
    :param max_length: Maximum length of a clickstream.
    :param vocab_size: Number of unique pages.
    :return: List of clickstreams.
    """
    clickstreams = []
    for _ in range(num_users):
        length = random.randint(5, max_length)
        clickstream = [random.randint(0, vocab_size - 1) for _ in range(length)]
        clickstreams.append(clickstream)
    return clickstreams

def train_model(model, dataloader, optimizer, criterion, num_epochs):
    """
    Train the clickstream model.
    :param model: The ClickstreamModel instance.
    :param dataloader: DataLoader for the training data.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param num_epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 100  # Number of unique pages
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 16
    num_epochs = 5
    learning_rate = 0.001

    # Generate dummy clickstream data
    num_users = 1000
    max_length = 20
    clickstreams = generate_dummy_clickstreams(num_users, max_length, vocab_size)

    # Prepare dataset and dataloader
    dataset = ClickstreamDataset(clickstreams)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: (
        torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True),
        torch.nn.utils.rnn.pad_sequence([item[1] for item in x], batch_first=True)
    ))

    # Initialize model, optimizer, and loss function
    model = ClickstreamModel(vocab_size, embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model(model, dataloader, optimizer, criterion, num_epochs)