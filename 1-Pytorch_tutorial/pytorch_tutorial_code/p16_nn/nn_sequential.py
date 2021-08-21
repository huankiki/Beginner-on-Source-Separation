import torch
import torchvision
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear = Linear(1024,10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# CIFAR10
dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

step = 0
test = myModel()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    img, target = data

    output = test(img)

    step = step + 1
    if step == 1:
        print(img.shape)
        print(output.shape)
    if step == 16:
        break
    # loss function
    result_loss = loss(output, target)
    print(result_loss)



# 显示NN网络graph
data = torch.ones([64,3,32,32])
writer = SummaryWriter("logs")
writer.add_graph(test, data)

writer.close()
