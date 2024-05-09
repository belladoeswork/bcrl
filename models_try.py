import torch
import torch.nn as nn
import torch.nn.functional as F

class TreatmentModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super(TreatmentModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        self.hidden_layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in self.hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.01))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size

        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(prev_size // 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(prev_size // 2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x