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
dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

step = 0
testModel = myModel()
loss = nn.CrossEntropyLoss()
#定义optim
optim = torch.optim.SGD(testModel.parameters(), lr=0.01)
for epoch in range(10):
    run_loss = 0.0
    for data in dataloader:
        img, target = data

        output = testModel(img)

        # step = step + 1
        # if step == 1:
        #     print(img.shape)
        #     print(output.shape)
        # if step >= 100:
        #     break

        # loss function
        result_loss = loss(output, target)
        # print(result_loss)

        # 学习模型
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        run_loss = run_loss + result_loss

    print(run_loss)

# tensor(1609.5360, grad_fn=<AddBackward0>)
# tensor(1349.5817, grad_fn=<AddBackward0>)
# tensor(1230.9514, grad_fn=<AddBackward0>)
# tensor(1157.8040, grad_fn=<AddBackward0>)
# tensor(1096.7056, grad_fn=<AddBackward0>)
# tensor(1041.5957, grad_fn=<AddBackward0>)
# tensor(991.2673, grad_fn=<AddBackward0>)

