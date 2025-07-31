import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Match Keras architecture: 150 -> 250 LSTM units
        if num_layers == 2:
            self.lstm1 = nn.LSTM(input_size, 150, 1, batch_first=True)
            self.lstm2 = nn.LSTM(150, 250, 1, batch_first=True)
        else:
            self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.lstm2 = None
        
        # Dropout after LSTM like Keras (0.5 in Keras)
        self.dropout = nn.Dropout(dropout)
        
        # Final prediction layer (no activation for regression)
        final_hidden = 250 if num_layers == 2 else hidden_size
        self.fc = nn.Linear(final_hidden, output_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for lstm in [self.lstm1, self.lstm2]:
            if lstm is not None:
                for name, param in lstm.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1 for better gradient flow
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize linear layer
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # First LSTM layer
        if self.lstm2 is not None:
            # Two-layer architecture like Keras: 150 -> 250
            lstm1_out, _ = self.lstm1(x)  # (batch, seq_len, 150)
            lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq_len, 250)
            # Take last time step
            last_output = lstm2_out[:, -1, :]  # (batch, 250)
        else:
            # Single or multi-layer LSTM
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            lstm_out, _ = self.lstm1(x, (h0, c0))
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Final prediction (linear output for regression)
        out = self.fc(last_output)
        return out 