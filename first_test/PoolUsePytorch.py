import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 使用nn.MaxPool2d()






# 1.将图转化为灰度图
im=Image.open('lena.jpg').convert('L')# 将图转化为灰度图
im=np.array(im,dtype='float32')# 将灰度图转化为矩阵
#plt.imshow(im.astype('uint8'), cmap='gray')
#plt.show()
# 2.将灰度图矩阵转化为tensor张量
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))

pool1 = nn.MaxPool2d(2, 2) # 池化时每一小块的大小是2*2
print('池化前的大小: :{} x {}'.format(im.shape[2], im.shape[3]))
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print('池化后的大小: : {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im1, cmap='gray')
plt.show()





