import torch
from torch import nn

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = input + 1
        return output

x = torch.tensor(1.0)
test = myModel()
output = test.forward(x)
print(output)


