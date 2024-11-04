import torch
import torch.nn as nn
import numpy as np
import random
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
    # Notice we don't need an "input size" or sequence length here.
    def __init__(self, vocab_size, hidden_size, num_layers, dropout_p = 0.5):
        super(ShakespeareRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html for embedding examples
        self.embedding = nn.Embedding(vocab_size, hidden_size) # could use an embedding size hyperparameter
        
        # self.recurrent = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True) # put it here
        self.recurrent = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.recurrent = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        #dropout layer to prevent overfirtting
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=1) #output logits

    # x will be a sequence of "n" input variables (letter indexes)
    # In pytorchtetilian 
    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.recurrent(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x, hidden

    def init_hidden(self, batch_size):
        # You need a hidden state for each RNN layer!
        
        ## LSTM
        # return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                # torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))

        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    

def train_model(model, data, epochs=100, seq_length=100, batch_size=64, lr=0.002, shuffle=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss() # we have a classification problem with logits

    batches = list(range(0, len(data) - seq_length, batch_size))

    for epoch in range(epochs):
        #maybe later iter randomly!
        if shuffle:
            random.shuffle(batches)
        
        with tqdm.tqdm(total=len(batches), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for i in batches:
                #ensure there's enough data tfor a full batch!
                if i + batch_size * seq_length >= len(data):
                    pbar.update(1)
                    continue
                
                #initiate hidden state
                hidden = model.init_hidden(batch_size)

                # Prepare input and target sequences
                
                # they are matrices with each input corresponding to output
                inputs = np.array([data[i+j:i+j+seq_length] for j in range(batch_size)])
                targets = np.array([data[i+j+1:i+j+seq_length+1] for j in range(batch_size)])

                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                targets = torch.tensor(targets, dtype=torch.long).to(device)

                #detach hidden state
                hidden = hidden.detach()

                # Forward pass
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                #update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        if (epoch + 1 % 10) == 0:
            torch.save(model,f'modelos/sk{epoch}.pth')

def generate_text(model, start_text, length=500, temperature = 1):
    #Not training model, don't need to save backpropagation values
    model.eval()

    #unsqueeze adds a batch dimension to the front of the input
    input_text = torch.tensor([char2idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    
    generated_text = start_text

    for _ in range(length):
        output, hidden = model(input_text, hidden)

        # use log probabilities for sampling
        log_probs = output[0,-1]

        # apply temperature
        log_probs /= temperature

        # convert to probabilities
        probabilities = torch.exp(log_probs)
        next_char_index = torch.multinomial(probabilities,1).item()

        # choose next char according to probabilties
        output_char = idx2char[next_char_index]
        

        # output_char = idx2char[torch.argmax(output[0, -1]).item()]
        generated_text += output_char

        #after generating the first character, we go one by one
        input_text = torch.tensor([char2idx[output_char]], dtype=torch.long).unsqueeze(0).to(device)

    return generated_text

#evaluation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
hidden_size = 512
num_layers = 3
epochs = 40
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
            shuffle=True)

# Generate text
print(generate_text(model, start_text="BRUTU",length=2000,temperature=0.1))
