import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=32, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer to convert character indices to vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Embed input
        embedded = self.embedding(x)
        
        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)
        
        # Get the last output
        output = output[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(output)
        
        return output, hidden


def prepare_data(text):
    """Prepare training data from text"""
    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create character to index and index to character mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create training sequences
    sequences = []
    targets = []
    
    # Create sequences
    for i in range(len(text) - 1):
        seq = text[:i+1]
        target = text[i+1]
        
        sequences.append([char_to_idx[ch] for ch in seq])
        targets.append(char_to_idx[target])
    
    return sequences, targets, char_to_idx, idx_to_char, vocab_size


def pad_sequences(sequences, max_len=None):
    """Pad sequences to the same length"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append([0] * (max_len - len(seq)) + seq)
        else:
            padded.append(seq[-max_len:])
    
    return padded


def train_model(text="hello world", epochs=500, learning_rate=0.01):
    """Train the RNN model"""
    print(f"Training on: '{text}'")
    print("=" * 50)
    
    # Prepare data
    sequences, targets, char_to_idx, idx_to_char, vocab_size = prepare_data(text)
    
    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, max_len)
    
    # Convert to tensors
    X = torch.LongTensor(padded_sequences)
    y = torch.LongTensor(targets)
    
    # Initialize model
    model = CharRNN(vocab_size, hidden_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs, _ = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    print("\nTraining complete!")
    print("=" * 50)
    
    return model, char_to_idx, idx_to_char, vocab_size


def predict_next_char(model, input_text, char_to_idx, idx_to_char, max_len=10):
    """Predict the next character given input text"""
    model.eval()
    
    with torch.no_grad():
        # Convert input text to indices
        try:
            input_indices = [char_to_idx[ch] for ch in input_text]
        except KeyError as e:
            return f"Error: Character {e} not in vocabulary"
        
        # Pad sequence
        if len(input_indices) < max_len:
            input_indices = [0] * (max_len - len(input_indices)) + input_indices
        else:
            input_indices = input_indices[-max_len:]
        
        # Convert to tensor
        input_tensor = torch.LongTensor([input_indices])
        
        # Get prediction
        output, _ = model(input_tensor)
        
        # Get probabilities
        probabilities = torch.softmax(output[0], dim=0)
        
        # Get predicted character
        predicted_idx = torch.argmax(probabilities).item()
        predicted_char = idx_to_char[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100
        
        return predicted_char, confidence, probabilities


def main():
    # Train the model
    text = "hello world"
    model, char_to_idx, idx_to_char, vocab_size = train_model(text, epochs=500)
    
    print("\n" + "=" * 50)
    print("Testing the model")
    print("=" * 50)
    
    # Test cases
    test_inputs = ["hell", "hel", "wor", "worl", "lo w", "o w", "h", "hello w"]
    
    max_len = len(text) - 1
    
    for test_input in test_inputs:
        if all(ch in char_to_idx for ch in test_input):
            predicted_char, confidence, probs = predict_next_char(
                model, test_input, char_to_idx, idx_to_char, max_len
            )
            print(f"\nInput: '{test_input}' → Predicted: '{predicted_char}' (Confidence: {confidence:.1f}%)")
            
            # Show top 3 predictions
            top_probs, top_indices = torch.topk(probs, min(3, vocab_size))
            print("  Top predictions:")
            for prob, idx in zip(top_probs, top_indices):
                char = idx_to_char[idx.item()]
                print(f"    '{char}': {prob.item()*100:.1f}%")
        else:
            print(f"\nInput: '{test_input}' → Contains characters not in training data")
    
    print("\n" + "=" * 50)
    print("Interactive mode - Enter your own sequences!")
    print("(Type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_input = input("\nEnter sequence: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        if all(ch in char_to_idx for ch in user_input):
            predicted_char, confidence, _ = predict_next_char(
                model, user_input, char_to_idx, idx_to_char, max_len
            )
            print(f"Predicted next character: '{predicted_char}' (Confidence: {confidence:.1f}%)")
        else:
            invalid_chars = [ch for ch in user_input if ch not in char_to_idx]
            print(f"Error: Characters {invalid_chars} not in vocabulary")
            print(f"Available characters: {list(char_to_idx.keys())}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()