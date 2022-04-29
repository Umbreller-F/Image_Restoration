import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


img = cv2.imread('Default.jpg')
cv2.imshow('origin', img)
cv2.waitKey(0)
print(img.shape)
# print(type(img))
img = np.random.rand(5,10,3)
print(img.shape)


def create_mask(origin, mask_rate):
    # mask_rate = 0.4 # 0.4/0.6/0.8
    # mask = np.random.randint(0)
    zeros_num = int(img.size * mask_rate)#根据0的比率来得到0的个数
    mask = np.ones(img.shape)#生成与原来模板相同的矩阵，全为1
    for i in range(mask.shape[0]):
        mask[i,:,]
    mask[:zeros_num] = 0 #将一部分换为0
    np.random.seed(16)
    np.random.shuffle(mask)#将0和1的顺序打乱
    mask = mask.reshape(img.shape)#重新定义矩阵的维度，与模板相同
    print(mask[:,:,0])
    return mask
# b,g,r = cv2.split(mask)

# print(b)
