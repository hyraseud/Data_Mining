import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size): ## This will also take care of initilizing the weights
        super(MyNetwork, self).__init__() 
        """
        In the constructor we instantiate one nn.Linear modules and assign it to f_x.
        This constructor has two parameters:
            first parameter: input dimension
            second parameter: output dimension
        """
        self.f_x = nn.Linear(input_size, output_size)
        self.output_size = 700

    def forward(self, x):
        y_hat = F.relu(self.f_x(x)) 
        return y_hat
    