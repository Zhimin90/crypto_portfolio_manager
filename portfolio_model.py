import torch
from torch import nn

# LSTM with no batch reset on hidden and cell state
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Initializing hidden state for first input with zeros
        self.h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_()
        # Initializing cell state for first input with zeros
        self.c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_()
        
    def forward(self, x):
        #print(x.shape)
        x = x.unsqueeze(0).unsqueeze(0)
        #print(x.shape)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (self.h0.detach(), self.c0.detach()))
        self.h0 = hn # Batches are in timely order. Hidden states can be passed thru batches.
        self.c0 = cn
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        out = out.squeeze(0)
        return out