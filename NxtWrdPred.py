import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "[UNK]"

    def preprocess_data(self, text):
        return text.lower().strip()

    def tokenize(self, sentence):
        sentence = self.preprocess_data(sentence)
        words = sentence.split()
        tokens = []

        for word in words:
            subword_tokens = []
            i = 0
            while i < len(word):
                found_match = False
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.vocab:
                        subword_tokens.append(subword)
                        i = j
                        found_match = True
                        break
                if not found_match:
                    for j in range(len(word), i, -1):
                        subword = f"##{word[i:j]}"
                        if subword in self.vocab:
                            subword_tokens.append(subword)
                            i = j
                            found_match = True
                            break
                if not found_match:
                    subword_tokens.append(self.unk_token)
                    break

            tokens.extend(subword_tokens)

        return tokens


class NeuralLMDataset(Dataset):
    def __init__(self, corpus_file, tokenizer, vocab, max_length=None):
        """
        Creates dataset for Neural LM training (next-word prediction task).
        """
        with open(corpus_file, "r") as f:
            self.corpus = f.readlines()

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.pad_token = "[PAD]"
        self.pad_idx = vocab.get(self.pad_token, 0)  # Default index for padding
        self.data = []

        self.max_length = max_length
        self.prepare_data()

    def preprocess_data(self, text):
        """Preprocess text by lowercasing and stripping spaces."""
        return text.lower().strip()

    def pad_sequence(self, sequence):
        """Pads or truncates the sequence to maintain uniform length across dataset."""
        if len(sequence) < self.max_length:
            sequence = [self.pad_idx] * (self.max_length - len(sequence)) + sequence
        else:
            sequence = sequence[:self.max_length]
        return sequence

    def prepare_data(self):
        """Tokenizes the corpus, creates (X, Y) pairs, and pads sequences."""
        all_inputs = []

        for sentence in self.corpus:
            sentence = self.preprocess_data(sentence)
            tokens = self.tokenizer.tokenize(sentence)
            indexed_tokens = [self.vocab[token] for token in tokens if token in self.vocab]

            # Generate (X, Y) pairs
            for i in range(1, len(indexed_tokens)):  # Start from 1 to ensure there's a label
                context = indexed_tokens[:i]  # Input sequence
                target = indexed_tokens[i]  # Next word token

                all_inputs.append(context)
                self.data.append((context, target))

        # Set max_length dynamically if not provided
        if self.max_length is None:
            self.max_length = max(len(x) for x in all_inputs)

        # Pad sequences
        self.data = [(self.pad_sequence(context), target) for context, target in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, max_length=10):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        context = context.to(device)
        embedded = self.embedding(context)
        embedded = embedded.view(embedded.shape[0], -1)
        hidden = F.relu(self.fc1(embedded))
        hidden = F.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        return output

class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, max_length=10):
        super(NeuralLM2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_length, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        context = context.to(device)
        embedded = self.embedding(context)
        embedded = embedded.view(embedded.shape[0], -1)
        hidden = F.relu(self.fc1(embedded))
        hidden = self.batchnorm1(hidden)
        hidden = self.dropout1(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.batchnorm2(hidden)
        hidden = self.dropout2(hidden)
        output = self.fc3(hidden)
        return output

class NeuralLM3(nn.Module):
    def __init__(self, vocab_size=14646, embedding_dim=64, hidden_dim=128, max_length=10):
        super(NeuralLM3, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim * max_length, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        context = context.to(device)

        embedded = self.embedding(context)
        embedded = embedded.view(embedded.shape[0], -1)

        hidden1 = F.relu(self.fc1(embedded))
        hidden1 = self.batchnorm1(hidden1)
        hidden1 = self.dropout1(hidden1)

        hidden2 = F.relu(self.fc2(hidden1))
        hidden2 = self.batchnorm2(hidden2)
        hidden2 = self.dropout2(hidden2)

        hidden3 = F.relu(self.fc3(hidden2))
        hidden3 = self.batchnorm3(hidden3)
        hidden3 = self.dropout3(hidden3)

        hidden3 = hidden3 + hidden1  # Skip connection (Residual Learning)

        output = self.fc4(hidden3)
        return output


# Load vocabulary
def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, token in enumerate(f.readlines()):
            vocab[token.strip()] = i
    return vocab

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_path):
    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            context , target = context.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                context , target = context.to(device),target.to(device)
                output = model(context)
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    # Plot Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {checkpoint_path}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{checkpoint_path}_loss.png")
    plt.show()

def compute_accuracy(model, data_loader):
    """
    Computes the accuracy of the model on a given dataset.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)  # (batch_size, vocab_size)
            predicted = torch.argmax(output, dim=1)  # Get token with highest probability
            correct += (predicted == target).sum().item()
            total += target.size(0)

    return correct / total

import numpy as np

def compute_perplexity(loss):
    """
    Computes the perplexity from the model's loss.
    """
    return np.exp(loss)

def evaluate_model(model, data_loader, criterion):
    """
    Computes accuracy and perplexity for a given dataset.
    """
    correct = 0
    total = 0
    total_loss = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for context, target in data_loader:
            context, target = context.to(device), target.to(device)
            output = model(context)  # (batch_size, vocab_size)
            loss = criterion(output, target)
            total_loss += loss.item()

            predicted = torch.argmax(output, dim=1)  # Get token with highest probability
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Compute Accuracy
    accuracy = correct / total

    # Compute Perplexity
    avg_loss = total_loss / len(data_loader)
    perplexity = compute_perplexity(avg_loss)

    return accuracy, perplexity

def predict_next_tokens(test_file, model, tokenizer, vocab, max_length):
    with open(test_file, "r") as f:
        sentences = f.readlines()
    
    model.to(device)
    model.eval()  # Set model to evaluation mode to avoid BatchNorm issues
    pad_idx = vocab.get("[PAD]", 0)  # Default pad index

    for sentence in sentences:
        sentence = sentence.strip()
        tokens = tokenizer.tokenize(sentence)
        indexed_tokens = [vocab[token] for token in tokens if token in vocab]

        # Pad sequence to max_length before prediction
        if len(indexed_tokens) < max_length:
            indexed_tokens = [pad_idx] * (max_length - len(indexed_tokens)) + indexed_tokens
        else:
            indexed_tokens = indexed_tokens[:max_length]

        predicted_tokens = []
        
        for _ in range(3):  # Predict up to 3 tokens
            context_tensor = torch.tensor(indexed_tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():  # Ensure gradients are not computed
                output = model(context_tensor)
            
            predicted_token_id = torch.argmax(output, dim=1).item()
            predicted_token = [token for token, idx in vocab.items() if idx == predicted_token_id][0]

            if predicted_token == "[PAD]" or predicted_token == "[UNK]":
                break  # Stop prediction if padding or unknown token is predicted

            predicted_tokens.append(predicted_token)
            indexed_tokens.append(predicted_token_id)

            # Keep only last `max_length` tokens for next prediction
            indexed_tokens = indexed_tokens[-max_length:]

        print(f"Input: {sentence.strip()}")
        print(f"Predicted Next Tokens: {' '.join(predicted_tokens)}\n")


def main():
    # Load the tokenizer (from WordPieceTokenizer implementation)
    vocab = load_vocab("vocabulary_18.txt")
    tokenizer = WordPieceTokenizer(vocab)
    test_file = "sample_test.txt"
    # Load dataset
    dataset = NeuralLMDataset("corpus.txt", tokenizer, vocab)

    # Split dataset into train (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Get max_length from dataset
    max_length = dataset.max_length

    # Train all models with dynamic `max_length`
    for i, model_class in enumerate([NeuralLM1, NeuralLM2, NeuralLM3], start=1):
        print(f"\nTraining Model NeuralLM{i}...\n")

        model = model_class(len(vocab), embedding_dim=64, hidden_dim=128, max_length=max_length)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        train(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, num_epochs=20, checkpoint_path=f"neural_lm{i}.pth")

    # Compute Accuracy & Perplexity for each model
    criterion = nn.CrossEntropyLoss()

    for i, model_class in enumerate([NeuralLM1, NeuralLM2, NeuralLM3], start=1):
        model_path = f"neural_lm{i}.pth"

        print(f"\nLoading {model_path} for evaluation...")

        # Load trained model
        model = model_class(len(vocab), embedding_dim=64, hidden_dim=128, max_length=max_length)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Evaluate Train & Validation Data
        train_accuracy, train_perplexity = evaluate_model(model, train_loader, criterion)
        val_accuracy, val_perplexity = evaluate_model(model, val_loader, criterion)

        print(f"Model NeuralLM{i}:")
        print(f"Train Accuracy: {train_accuracy:.4f}, Train Perplexity: {train_perplexity:.2f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Perplexity: {val_perplexity:.2f}\n")

    max_length = dataset.max_length  # Make sure this matches dataset processing
    model = NeuralLM3(len(vocab), embedding_dim=64, hidden_dim=128, max_length=max_length)
    # Run predictions on test file
    predict_next_tokens(test_file, model, tokenizer, vocab, max_length)

if __name__ == "__main__":
    main()