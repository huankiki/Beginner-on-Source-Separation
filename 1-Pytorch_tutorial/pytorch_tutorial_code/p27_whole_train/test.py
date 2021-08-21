import torch
import torchvision
from PIL import Image
from model import myModel

class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

img_path = "../dataset/test/dog.png"
image = Image.open(img_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
print(image)
image = transform(image)
print(image.shape)

model = torch.load("mymodel_cifar.pth", map_location=torch.device('cpu'))
# print(model)
image = torch.reshape(image, (1,3,32,32))

model.eval()
with torch.no_grad():
    output = model(image)

print(output)

pred = output.argmax(1)
print(pred)
