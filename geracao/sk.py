import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
import tqdm

# Load text data
with open('shakespeare.txt', 'r') as file:
    text = file.read()

# Determine device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Selected device:{device}')

# Create character-to-index and index-to-character mappings
chars = sorted(set(text))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for idx, ch in enumerate(chars)}

# Encode text as integers
text_encoded = np.array([char2idx[ch] for ch in text], dtype=np.int64)
vocab_size = len(chars)

class ShakespeareRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(ShakespeareRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=0.5,batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def train_model(model, data, epochs=100, seq_length=100, batch_size=64, lr=0.002):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hidden = model.init_hidden(batch_size).to(device)

    for epoch in range(epochs):
        #(len(data) - seq_length) // batch_size
        with tqdm.tqdm(total=1000 , desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            # for i in range(0, len(data) - seq_length, batch_size):
            for i in range(0,1000*batch_size,batch_size):
                if i + seq_length*batch_size > len(data) : 
                    pbar.update(1)
                    continue

                inputs = np.array([data[i+j:i+j+seq_length] for j in range(batch_size)])
                targets = np.array([data[i+j+1:i+j+seq_length+1] for j in range(batch_size)])
                # Prepare input and target sequences
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                targets = torch.tensor(targets, dtype=torch.long).to(device)

                hidden = hidden.detach()
                # Forward pass
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())


def generate_text(model, start_text, length=500, temperature=1.0):
    model.eval()
    input_text = torch.tensor([char2idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1).to(device)

    generated_text = start_text

    for _ in range(length):
        output, hidden = model(input_text, hidden)
        hidden.detach()

        # Scale logits by temperature
        logits = output[0, -1] / temperature
        probabilities = torch.nn.functional.softmax(logits, dim=0)

        # Sample from the scaled distribution
        output_char_idx = torch.multinomial(probabilities, 1).item()
        output_char = idx2char[output_char_idx]
        generated_text += output_char

        input_text = torch.tensor([[output_char_idx]], dtype=torch.long).to(device)

    return generated_text

#evaluation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
hidden_size = 512
num_layers = 2
epochs = 1
seq_length = 100
batch_size = 64
learning_rate = 0.002

# instantiate the model
model = ShakespeareRNN(vocab_size, hidden_size, num_layers).to(device)
# model = ShakespeareRNN(vocab_size, hidden_size, num_layers).cuda()

# Train the model
train_model(model, 
            text_encoded, 
            epochs=epochs, 
            seq_length=seq_length, 
            batch_size=batch_size,
            lr=learning_rate,
            )

# Generate text
print(generate_text(model, start_text="BRU",length=2000))
