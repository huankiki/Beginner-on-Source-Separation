import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
# test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
# test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

step = 0
writer = SummaryWriter("logs")
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)

    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()

