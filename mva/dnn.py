#This module contains the definition for a DNN
import torch
import torch.nn as nn

class ObjectNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ObjectNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size)
        self.batchnorm4 = nn.BatchNorm1d(hidden_size)

        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        self.layernorm4 = nn.LayerNorm(hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        #Using batch normalization
        #x = self.relu(self.batchnorm1(self.fc1(x)))
        #x = self.relu(self.batchnorm2(self.fc2(x)))
        #x = self.relu(self.batchnorm3(self.fc3(x)))
        #x = self.relu(self.batchnorm4(self.fc4(x)))

        #Using layer normalization
        x = self.relu(self.layernorm1(self.fc1(x)))
        x = self.relu(self.layernorm2(self.fc2(x)))
        x = self.relu(self.layernorm3(self.fc3(x)))
        x = self.relu(self.layernorm4(self.fc4(x)))
        x = self.output_layer(x)
        return x

