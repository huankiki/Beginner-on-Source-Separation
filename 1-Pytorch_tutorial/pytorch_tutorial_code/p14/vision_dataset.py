import torchvision
from torch.utils.tensorboard import SummaryWriter

# train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True)
#
# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes)
# print(test_set.classes[target])
# img.show(img)

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=dataset_transform, download=False)
print(test_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
