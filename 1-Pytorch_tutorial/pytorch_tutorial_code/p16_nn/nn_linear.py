import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x


# CIFAR10
dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

step = 0
test = myModel()
for data in dataloader:
    img, target = data
    # reshape 1
    # img = torch.reshape(img, (1,1,1,-1))
    # reshape 2
    # print(img.shape)
    img = torch.flatten(img)
    # print(img.shape)

    output = test(img)
    step = step + 1
    if step == 1:
        print(img.shape)
        print(output.shape)
    if step == 16:
        break

