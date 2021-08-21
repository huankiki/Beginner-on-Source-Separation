import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]], dtype=torch.float32)
print(input.shape)
input = torch.reshape(input, (-1,1,5,5))
print(input.shape)

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=(3,3), ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

test = myModel()
output = test(input)
print(output)

# CIFAR10
dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=4)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    img, target = data
    output = test(img)
    step = step + 1
    writer.add_images("input_maxPool", img, step)
    writer.add_images("output_maxPool", output, step)

    if step == 1:
        print(img.shape)
        print(output.shape)
    if step == 16:
        break

writer.close()