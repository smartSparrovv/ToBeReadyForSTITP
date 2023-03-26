import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im=Image.open('lena.jpg').convert('L')# 将图转化为灰度图
# plt.imshow(im, cmap='gray')
# plt.show()

# 池化，就是按一小块一小块地选取图上的矩阵的运算结果，若是最大池化，运算就是选取一小块里最大的数字；平均池化就是计算一小块的平均值
# 和卷积不同，池化只需要原图即可，没有类似卷积核的东西
# 池化的效果是压缩图片
# 最大池化np.max(x)，均值池化np.sum(x)/(n*m)，最小池化np.min(x)
# data是原图用np.array(原图)生成的矩阵,m和n分别是池化的每一小块的行数和列数
def pooling(data, m, n):
    a,b = data.shape # a、b分别是原图的行和列
    img_new = []
    for i in range(0,a,m): #行上的步长是m，即:若第0行的池化完后，下一次从第0+m行开始
        line = []#记录每一行
        for j in range(0,b,n): #列上的步长是n，即:若第0列的池化完后，下一次从第0+n列开始
            x = data[i:i+m,j:j+n]#选取池化区域\n",
            line.append(np.max(x))   #将一行上池化每一小块的结果存入一维数组里
        img_new.append(line)  # 将每一行的池化结果存入二维数组
    return np.array(img_new)

im_new = pooling(np.array(im),2,2)
plt.imshow(im_new,cmap='gray')
plt.show()

