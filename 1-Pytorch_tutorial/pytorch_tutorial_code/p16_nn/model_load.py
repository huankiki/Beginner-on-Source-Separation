import torch
import torchvision

# load method 1
# model = torch.load("vgg16_method1.pth")
# print(model)

# load method 2, need load nn model class too
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# model = torch.load("vgg16_method2.pth")
# print(model)
