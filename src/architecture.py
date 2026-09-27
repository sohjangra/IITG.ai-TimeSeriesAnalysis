import torch
import torch.nn as nn

class DualBranchVolatilityNet(nn.Module):
    """
    Dual-Branch network for volatility forecasting.
    Combines spatial shock extraction (1D-CNN) and temporal memory (LSTM) via gated attention.
    """
    def __init__(self, num_features, seq_len=78, hidden_dim=64):
        super(DualBranchVolatilityNet, self).__init__()
        
        # Branch A: 1D-CNN extracts price shocks and spatial regime breaks
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        cnn_out_seq = seq_len // 2 // 2
        self.cnn_flatten_dim = 64 * cnn_out_seq
        self.cnn_fc = nn.Linear(self.cnn_flatten_dim, hidden_dim)
        
        # Branch B: 2-layer LSTM models historical volatility persistence
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Gated attention dynamically balances CNN shocks vs LSTM memory
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Softplus guarantees non-negative predicted volatility
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        c = self.relu(self.conv1(x_cnn))
        c = self.pool(c)
        c = self.relu(self.conv2(c))
        c = self.pool(c)
        c = c.view(c.size(0), -1)
        cnn_features = self.relu(self.cnn_fc(c))
        
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        # Dynamically blend branches via attention gate
        combined = torch.cat((cnn_features, lstm_features), dim=1)
        gate = self.attention_gate(combined)
        mixed_features = gate * cnn_features + (1 - gate) * lstm_features
        
        predicted_vol = self.fc_out(mixed_features)
        return predicted_vol, gate


class QLIKELoss(nn.Module):
    """QLIKE loss penalizes under-predicting volatility to prevent crash liquidation."""
    def __init__(self, eps=1e-8):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred + self.eps
        y_true = y_true + self.eps
        loss = (y_true / y_pred) - torch.log(y_true / y_pred) - 1
        return torch.mean(loss)


if __name__ == "__main__":
    batch_size, seq_len, num_features = 16, 78, 10
    mock_input = torch.randn(batch_size, seq_len, num_features)
    mock_target = torch.rand(batch_size, 1) * 0.05
    
    model = DualBranchVolatilityNet(num_features=num_features, seq_len=seq_len)
    criterion = QLIKELoss()
    predictions, gate_weights = model(mock_input)
    loss = criterion(predictions, mock_target)
    
    print(f"Model initialized. Sample loss: {loss.item():.4f}")