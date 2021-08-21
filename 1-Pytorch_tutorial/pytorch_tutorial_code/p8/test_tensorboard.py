from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# conda install tensorboard
# 注释快捷键：Ctrl+/
# pip install opencv-python

writer = SummaryWriter("logs")


# img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
# Image.open()的type类型：<class 'PIL.JpegImagePlugin.JpegImageFile'>
# 需要转换成Tensor或者numpy.array
image_path = "../dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)  #(512, 768, 3)
img_PIL.show()
# 对Tensor的Shape有要求
# img_tensor: Default is :math: (3, H, W)

writer.add_image("test", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
