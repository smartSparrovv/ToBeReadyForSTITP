import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)


im = Image.open('lena.jpg').convert('L')
im_new = relu(im)
plt.imshow(im_new,cmap='gray')
plt.show()