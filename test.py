import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#定义二维傅里叶变换(f是待变换的二维矩阵)
def two_dimensional_fourier_transform(f):
    x_len, y_len = f.shape
    F = np.zeros((x_len, y_len), dtype=complex)
    for u in range(x_len):
        for v in range(y_len):
            sum_val = 0+0j
            for x in range(x_len):
                for y in range(y_len):
                    exponent = -2j * np.pi * ((u * x / x_len) + (v * y / y_len))#课本p6 式1.3.6 Forier变换式
                    sum_val += f[x, y] * np.exp(exponent)
            F[u, v] = sum_val
    return F


#定义二维逆傅里叶变换(f是待逆变换的二维矩阵)
def inverse_two_dimensional_fourier_transform(f):
    x_len, y_len = f.shape
    f = np.zeros((x_len, y_len), dtype=complex)
    for x in range(x_len):
        for y in range(y_len):
            sum_val = 0+0j
            for u in range(x_len):
                for v in range(y_len):
                    exponent = 2j * np.pi * ((u * x / x_len) + (v * y / y_len))#课本p6 式1.3.7 逆Forier变换式
                    sum_val += f[u, v] * np.exp(exponent)
            f[x, y] = sum_val / (x_len * y_len)
    return f


#-----调用-----
#读取图片，获得灰度图矩阵
img = Image.open("cat.jpg").convert("L")  
img_array = np.array(img)

F=two_dimensional_fourier_transform(img_array)
f=inverse_two_dimensional_fourier_transform(F)

#显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_array, cmap='gray')  
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Fourier Transform Magnitude")
plt.imshow(np.log(np.abs(F) + 1), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(np.abs(f), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

