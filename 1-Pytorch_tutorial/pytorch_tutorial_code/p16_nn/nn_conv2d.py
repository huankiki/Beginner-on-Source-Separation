import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3), stride=(1,1), padding=(0,0))

    def forward(self, x):
        x = self.conv1(x)
        return x

test = myModel()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    img, target = data
    output = test(img)
    step = step + 1
    writer.add_images("input", img, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    if step == 1:
        print(img.shape)    #torch.Size([64, 3, 32, 32])
        print(output.shape) #torch.Size([64, 6, 30, 30])
    if step == 16:
        break

writer.close()