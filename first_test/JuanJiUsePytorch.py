import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
# 1.将图转化为灰度图
im=Image.open('lena.jpg').convert('L')# 将图转化为灰度图
im=np.array(im,dtype='float32')# 将灰度图转化为矩阵
#plt.imshow(im.astype('uint8'), cmap='gray')
#plt.show()

# 2.将灰度图转化为tensor张量
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
# 上句，将灰度图转化为tensor张量，第一个1是图片数，第二个1是图片的通道数，灰度图的通道数是1，后两个参数分别是图片的高和宽，即矩阵的行和列
conv1 = nn.Conv2d(1, 1, 3, bias=False) # 定义卷积
# 定义轮廓检测卷积核!!!!!!!!!!!!!
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],dtype='float32')
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) # 适配卷积的输入输出
# 上句，第一个1是卷积核的组数，第二个是卷积核的通道数，后两个是卷积核的大小为3*3.这是为了适配图im
# 给卷积的 kernel 赋值
conv1.weight.data = torch.from_numpy(sobel_kernel)# 将轮廓检测卷积核赋给定义好的卷积
edge1 = conv1(Variable(im)) # 作用在图片上,edge1就是卷积之后的结果
edge_1 = edge1.data.squeeze().numpy() # 将输出转换为图片的格式,其实就是将tensor张量类型变成numpy类型
#plt.imshow(edge_1, cmap='gray') #输出结果
#plt.show()


im_1=im.data.squeeze().numpy()
plt.figure()  #执行一次此语句，则出现一个新的图片显示窗口,但要把plt.show()写在最后才能让多个图一起出现
plt.imshow(im_1, cmap='gray')
plt.figure()
plt.imshow(edge_1, cmap='gray')
plt.show()