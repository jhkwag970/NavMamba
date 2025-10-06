import torch
import torch.nn as nn

class NavLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.imu_rnn = nn.LSTM(6, hidden_size, num_layers, batch_first=True)
        self.gnss_fc = nn.Sequential(nn.Linear(3, hidden_size), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, imu_seq, gnss_seq):
        imu_out, _ = self.imu_rnn(imu_seq)          # [B, T, H]
        gnss_feat = self.gnss_fc(gnss_seq)          # [B, T, H]
        fused = torch.cat([imu_out, gnss_feat], dim=-1)
        return self.fc(fused)                       # [B, T, 3]