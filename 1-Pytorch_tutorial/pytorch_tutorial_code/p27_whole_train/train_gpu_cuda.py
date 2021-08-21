import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import myModel
import time

#-------------------------------------------
#model, loss, nn.input, nn.target 有cuda()
#dataloader, optimizer 没有cuda()
#-------------------------------------------

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度： {}".format(train_data_size))
print("测试数据集的长度： {}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloder = DataLoader(train_data, batch_size=64)
test_dataloder = DataLoader(test_data, batch_size=64)

# 创建网络模型
mymodel = myModel()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    mymodel.cuda()
    loss_fn.cuda()
# 优化器
learn_rate = 0.001
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learn_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("logs")
start_time = time.time()
for i in range(epoch):
    print("--------第 {} 轮训练开始 --------".format(i+1))
    #训练步骤开始
    mymodel.train()
    for data in train_dataloder:
        img, target = data
        if torch.cuda.is_available():
            img.cuda()
            target.cuda()
        output = mymodel(img)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(end_time-start_time)
    # 测试步骤
    total_test_loss = 0
    total_accuracy = 0
    mymodel.eval()
    with torch.no_grad():
        for data in test_dataloder:
            img, target = data
            if torch.cuda.is_available():
                img.cuda()
                target.cuda()
            output = mymodel(img)
            loss = loss_fn(output, target)
            total_test_loss = total_test_loss + loss
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集的loss: {}".format(total_test_loss))
    print("整体测试集的正确率: {}".format(total_accuracy/test_data_size))
    total_test_step = total_test_step + 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

writer.close()

torch.save(mymodel, 'mymodel_cifar.pth')

