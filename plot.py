import cv2
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]

def original(i, j, k, ksize, img):
    # 找到矩阵坐标
    x1 = y1 = -ksize // 2
    x2 = y2 = ksize + x1
    temp = np.zeros(ksize * ksize)
    count = 0
    # 处理图像
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return temp

# 自定义最大值滤波器最小值滤波器
def max_min_functin(ksize, img, flag):
    img0 = img.copy()
    img = img.copy()
    for i in range(0, img0.shape[0]):
        for j in range(2, img0.shape[1]):
            for k in range(img0.shape[2]):
                temp = original(i, j, k, ksize, img0)
                if flag == 0:
                    img[i, j, k] = np.max(temp)
                elif flag == 1:
                    img[i, j, k] = np.min(temp)
    return img

def plot1():

    fig, axes = plt.subplots(2,3,figsize=(10,8))
    # 清晰图像
    image1 = cv2.imread("/Users/wangheng/workspace/pycharmworkspace/defog/FFA-Net/fig/0099_0_FFA.png")
    image1 = cv2.resize(image1,(300, 400))
    image2 = np.min(image1, axis=-1)
    image3 = max_min_functin(3, image1, 1) # 最小值滤波图像

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    axes[0,0].imshow(image1)
    axes[0,0].set_xlabel('清晰图像RGB')
    axes[0,1].imshow(image2)
    axes[0,1].set_xlabel('清晰图像-暗通道')
    axes[0,2].imshow(image3)
    axes[0,2].set_xlabel('清晰图像-最小滤波')


    # hazy 图像
    image1 = cv2.imread("/Users/wangheng/workspace/pycharmworkspace/defog/FFA-Net/fig/0099_0.9_0.16.jpg")
    image1 = cv2.resize(image1,(300, 400))
    image2 = np.min(image1, axis=-1)
    image3 = max_min_functin(5, image1, 1) # 最小值滤波图像

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    axes[1,0].imshow(image1)
    axes[1,0].set_xlabel('有雾图像RGB')
    axes[1,1].imshow(image2)
    axes[1,1].set_xlabel('有雾图像-暗通道')
    axes[1,2].imshow(image3)
    axes[1,2].set_xlabel('有雾图像-最小滤波')

    for i in range(2):
        for j in range(3):
            axes[i,j].get_yaxis().set_visible(False)
            axes[i,j].set_xticks([])

    plt.tight_layout()
    plt.show()
    fig.savefig('./exp_images/1.png', format='png', dpi=600)


if __name__ == '__main__':
    plot1()