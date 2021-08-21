import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


image_path = "../dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_PIL)
# img_PIL.show()

# 1-transforms.ToTensor
tensor = transforms.ToTensor()
tensor_img = tensor(img_PIL)
print(type(tensor_img))

# PIL to tensor
writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()
