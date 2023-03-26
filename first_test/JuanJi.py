from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('lena.jpg')
plt.axis("off")
plt.imshow(img)  # 用于接受和处理图像 但无法显示图像
plt.show()  # 用于展示图像，将plt.imshow()处理的图像进行展示
gray = img.convert('L')  # 将img转化为灰度图
plt.figure()
plt.imshow(gray, cmap='gray')
plt.show()

# 2、将RGB三通道数据分离，三色通道可视化
r, g, b = img.split()
plt.figure()
plt.subplot(131)  # 创建单个子图，该图在一行三列里的第一个
plt.imshow(r)
plt.subplot(132)
plt.imshow(g)
plt.subplot(133)
plt.imshow(b)
plt.show()

# 3、对单通道的数据进行卷积，如r,g,b,或者gray,将数据先转换为numpy类型，获取数据维度\n",
# 4、定义卷积核，如高斯滤波\n",
data = np.array(gray)  # 将图像转化为矩阵
n, m = data.shape
# 定义一个卷积和，下面这个是高斯滤波
k = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


# 5、定义卷积操作函数
def convolution(k, data):
    n, m = data.shape  # n为data的行数，m为data的列数
    img_new = []  # 该数组存储结果，是个二维数组
    for i in range(n - 3):  # 有n行，但因为核的大小是3*3，所以只需要从0行到n-3行
        line = []  # 该数组存储对每一行卷积的结果，是一维数组
        for j in range(m - 3):  # 有m行，但因为核的大小是3*3，所以只需要从0行到m-3列
            a = data[i:i + 3, j:j + 3]  # 得到data上一个个3*3的小块
            line.append(np.sum(np.multiply(k,a)))  # 让小块和卷积核运算，方式是:对应元素相乘，并将结果相加。结果存入数组line。line最终变成该行卷积的结果。        img_new.append(line)
        img_new.append(line)  # 将每一行卷积的结果存入数组img_new,最终变成一个卷积结果的二维数组
    return np.array(img_new)

# 6、调用卷积函数
img_new = convolution(k, data)#卷积过程
plt.imshow(img_new, cmap='gray')
plt.axis('off')
plt.show()

# 7、可视化卷积结果
print("final")