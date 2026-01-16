#!/usr/bin/env python3
import torch
import torch.nn as nn

class CowActivityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout_prob: float = 0.5):
        """
        LSTM for cow activity classification.
        Args:
          input_size   : 3 (ax, ay, az)
          hidden_size  : hidden units
          num_layers   : LSTM layers
          num_classes  : number of behaviors
          dropout_prob : dropout (applied between LSTM layers and on head)
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))   # (B,T,H)
        out = out[:, -1, :]               # last time step
        out = self.dropout(out)
        out = self.fc(out)                # (B,num_classes)
        return out

