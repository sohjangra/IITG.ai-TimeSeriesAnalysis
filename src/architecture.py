import torch
import torch.nn as nn

class DualBranchVolatilityNet(nn.Module):
    """
    Dual-Branch State-Switching Network for Volatility Prediction.
    Branch A (CNN): Extracts sharp structural regime shifts & shocks.
    Branch B (LSTM): Captures long-term historical volatility memory.
    """
    def __init__(self, num_features, seq_len=78, hidden_dim=64):
        super(DualBranchVolatilityNet, self).__init__()
        
        # ==========================================
        # Branch A: 1D-CNN (Spatial Shock Filter)
        # ==========================================
        # PyTorch Conv1d expects input shape: (Batch, Channels, Seq_Len)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Calculate the sequence length after two MaxPool1d(2) layers
        cnn_out_seq = seq_len // 2 // 2
        self.cnn_flatten_dim = 64 * cnn_out_seq
        
        # Project CNN output to a standard hidden dimension
        self.cnn_fc = nn.Linear(self.cnn_flatten_dim, hidden_dim)
        
        # ==========================================
        # Branch B: LSTM (Temporal Memory)
        # ==========================================
        # PyTorch LSTM expects input shape: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        # ==========================================
        # Gated Attention Mechanism
        # ==========================================
        # Learns to weight CNN vs LSTM based on the current context 
        # (e.g., Variance Ratio spikes / structural breaks)
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Squashes output to a weight between 0 and 1
        )
        
        # ==========================================
        # Final Output Layer
        # ==========================================
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Ensures predicted volatility is strictly positive
        )

    def forward(self, x):
        """
        Forward pass for the network.
        Input 'x' shape: (Batch, Seq_Len, Num_Features)
        """
        # --- Branch A Forward ---
        # Permute input for CNN: (Batch, Channels/Features, Seq_Len)
        x_cnn = x.permute(0, 2, 1) 
        
        c = self.relu(self.conv1(x_cnn))
        c = self.pool(c)
        c = self.relu(self.conv2(c))
        c = self.pool(c)
        
        c = c.view(c.size(0), -1) # Flatten spatial dimensions
        cnn_features = self.relu(self.cnn_fc(c)) # Shape: (Batch, hidden_dim)
        
        # --- Branch B Forward ---
        lstm_out, _ = self.lstm(x)
        # Extract the hidden state from the final time step of the sequence
        lstm_features = lstm_out[:, -1, :] # Shape: (Batch, hidden_dim)
        
        # --- Gated Attention Merger ---
        # Concatenate features to let the network determine the current market state
        combined = torch.cat((cnn_features, lstm_features), dim=1)
        
        # Gate output alpha approaches 1 during shocks (favoring CNN), 
        # and approaches 0 during mean-reversion (favoring LSTM)
        gate = self.attention_gate(combined) 
        
        # Weighted combination of both branches
        mixed_features = gate * cnn_features + (1 - gate) * lstm_features
        
        # --- Final Volatility Prediction ---
        predicted_vol = self.fc_out(mixed_features)
        
        # We return both the prediction and the gate value. 
        # The gate value can be logged during live execution to monitor market regime!
        return predicted_vol, gate


class QLIKELoss(nn.Module):
    """
    Custom QLIKE Loss Function.
    Heavily penalizes the model when it underpredicts realized volatility/risk.
    """
    def __init__(self, eps=1e-8):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure positivity to prevent log(0) errors
        y_pred = y_pred + self.eps
        y_true = y_true + self.eps
        
        # QLIKE Formula: y_true / y_pred - log(y_true / y_pred) - 1
        loss = (y_true / y_pred) - torch.log(y_true / y_pred) - 1
        
        return torch.mean(loss)

# ==========================================
# Example Usage / Sanity Check
# ==========================================
if __name__ == "__main__":
    # Simulate a batch of 16 sequences, 78 5-min bars, 10 engineered features
    batch_size = 16
    seq_len = 78
    num_features = 10
    
    mock_input = torch.randn(batch_size, seq_len, num_features)
    mock_target = torch.rand(batch_size, 1) * 0.05 # Mock positive volatility targets
    
    # Initialize Model and Loss
    model = DualBranchVolatilityNet(num_features=num_features, seq_len=seq_len)
    criterion = QLIKELoss()
    
    # Forward Pass
    predictions, gate_weights = model(mock_input)
    
    # Calculate Loss
    loss = criterion(predictions, mock_target)
    
    print(f"Model Architecture Initialized Successfully.")
    print(f"Input Shape: {mock_input.shape}")
    print(f"Prediction Shape: {predictions.shape}")
    print(f"Gate Weights Shape: {gate_weights.shape}")
    print(f"Sample QLIKE Loss: {loss.item():.4f}")