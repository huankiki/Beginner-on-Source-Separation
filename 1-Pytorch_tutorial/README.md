
## Install Anaconda & Pytorch
可参考：[ref1](https://blog.csdn.net/weixin_44789149/article/details/109504715)，[ref2](https://blog.csdn.net/zzq060143/article/details/88042075)，[ref3](https://blog.csdn.net/weixin_41608328/article/details/103986181)


- **1，下载和安装anaconda**
- **2，添加环境变量**

把anaconda的安装目录的Scripts文件夹路径，添加到“控制面板\系统和安全\系统\高级系统设置\环境变量\用户变量\PATH”。
以管理员的身份运行cmd命令行，输入`conda --version`，有返回结果表示安装成功。
- **3，创建虚拟环境**

一般做项目都要用虚拟环境，`conda env list`可以列出当前conda的所有虚拟环境。

`conda create -n virenv_name`：创建做项目的虚拟环境。

`conda activate pytorch_learn`：切换到虚拟环境。

`conda list`：列出当前的package。

`conda install <pkg_name>`：用conda安装package。
- **4，设置清华源，安装pytorch**

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda install pytorch
```


## Pytorch Tutorial
[B站： PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】](https://www.bilibili.com/video/BV1hE411t7RN)


- `help` & `dir`（P4)

```
import torch
dir(torch)
dir(torch.cuda)
help(torch.cuda.is_available())
```

注释快捷键：`Ctrl+/`

- Dataset / Dataloader （P6/P7)

`Dataset`：提供一种方式去获取数据及其label

`Dataloader`：为网络提供不同的数据形式

- Tensorboard（P8/P9)

安装tensorboard：`conda install tensorboard`

用命令打开logs：`tensorboard --logdir=logs  <--port=6007>`

```
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()
```

- transform of torchvision（P10/11)

`from torchvision import transforms`


关注输入、输出，多看[官方文档](https://pytorch.org/vision/stable/index.html)

- vision数据集（P14)
```
import torchvision
train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
```

- 神经网络（P16-P23)
```
from torch import nn
from torch.nn import Conv2d
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3), stride=(1,1), padding=(0,0))

    def forward(self, x):
        x = self.conv1(x)
        return x
```
[理解：super(XXX, self).__init__()](https://blog.csdn.net/dongjinkun/article/details/114575998)

```
nn.Module，Containers，Base class for all neural network modules.
nn.Conv2d，Convolution Layers
nn.MaxPool2d，Pooling Layers
nn.ReLU，Non-linear activations
nn.BatchNorm2d，Normalization Layers
nn.LSTM，Recurrent Layers
nn.Linear，Linear Layers
nn.Sequential，Sequential
nn.MSELoss，nn.CrossEntropyLoss，Loss Functions
```

- torch.optim（P24)

[优化器官网介绍](https://pytorch.org/docs/stable/optim.html)
```
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```


- 模型的保存与读取（P26)
```
vgg16 = torchvision.models.vgg16(pretrained=False)

# save method 1
torch.save(vgg16, "vgg16_method1.pth")

# save method 2
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# load method 1
model = torch.load("vgg16_method1.pth")
# print(model)

# load method 2, need load nn model class too
model = torch.load("vgg16_method2.pth")
```


- 完整的模型训练+GPU训练（P27/28/29，P30/31)
- 完整的模型测试（P32）


