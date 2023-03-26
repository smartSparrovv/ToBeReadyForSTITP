import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 第一步，先把原图转化为灰度图
im=Image.open('lena.jpg').convert('L')# 将图转化为灰度图
im=np.array(im,dtype='float32')# 将灰度图转化为矩阵

# 第二步，将numpy矩阵转化为tensor张量
im=torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))

# 第二步，对灰度图的tensor张量卷积
conv1 = nn.Conv2d(1, 1, 3, bias=False) # 定义卷积
# 定义轮廓检测卷积核!!!!!!!!!!!!!
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],dtype='float32')
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) # 适配卷积的输入输出
# 上句，第一个1是卷积核的组数，第二个是卷积核的通道数，后两个是卷积核的大小为3*3.这是为了适配图im
# 给卷积的 kernel 赋值
conv1.weight.data = torch.from_numpy(sobel_kernel)# 将轮廓检测卷积核赋给定义好的卷积
edge1 = conv1(Variable(im)) # 作用在图片上,edge1就是卷积之后的结果

# 第三步，对卷积结果池化
pool1 = nn.MaxPool2d(2, 2)
smallIm = pool1(edge1)

# 第四步，对池化后的卷积结果进行激活
x=F.relu(edge1)
x=x.data.squeeze().numpy()
plt.figure()
plt.imshow(x,cmap='gray')
plt.show()