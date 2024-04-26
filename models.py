# import torch
# import torch.nn as nn

# class TreatmentModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(TreatmentModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# models.py version 1.0

import torch
import torch.nn as nn

class TreatmentModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TreatmentModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    