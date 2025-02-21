# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

"""
    numpy实现一元线性回归(梯度下降法)  假设 y=wx+b
"""

def generate_data(num):
    enlarge = 30  
    x = np.random.rand(num) * enlarge   # 放大适当的倍数, 减小噪声数据的影响
    noise = np.random.randn(num)
    y = 0.6 * x + noise    
    return x, y
    
    
def update_parameter(x, w, b, lr):
    w_grad = 0
    b_grad = 0
    n = len(x)
    for i in range(n):
        # 单个样本损失: loss = (y_pred - y) ** 2 = (wx + b - y) ** 2
        # loss对w的偏导数
        w_grad += 2 * (w * x[i] + b - y[i]) * x[i]
        # loss对b的偏导数
        b_grad += 2 * (w * x[i] + b - y[i]) * 1
    w -= lr * w_grad / n
    b -= lr * b_grad / n
    return w, b


def draw(x, y, y_pred, final=True):
    plt.clf()
    plt.scatter(x, y, c="blue")
    plt.plot(x, y_pred, c="red")
    plt.pause(0.5) if final else plt.show()

        
def train(x, y, w, b, lr, epoch):
    plt.ion()
    # 储存历史损失
    loss_list = []
    for i in range(epoch):
        # 用当前参数计算y_pred
        y_pred = w * x + b
        # 计算所有样本平均损失
        loss = ((y - y_pred) ** 2).sum() / len(y)
        loss_list.append(loss)
        logger.warning((f"epoch:{i+1},  loss:{loss:.4f},  w:{w},  b:{b}"))
        # 更新参数
        w, b = update_parameter(x, w, b, lr)
        # 用新参数计算y_pred, 并画图
        y_pred = w * x + b
        draw(x, y, y_pred)
    plt.ioff()
    draw(x, y, y_pred, False)
    return loss_list


if __name__ == "__main__":
    
    # 设置随机种子,生成数据
    np.random.seed(2022)
    x, y = generate_data(num=60)
    
    PLOT = False
    if PLOT:
        # 查看数据
        plt.scatter(x, y, c="blue")
        plt.show()
    else:
        # 设置随机初始参数
        w, b = np.random.randn(1), np.random.randn(1)
        # 设置较小的学习率
        lr = 0.0005
        # 设置训练轮次
        epoch = 20
        # 训练数据
        loss_list = train(x, y, w, b, lr, epoch)
        # 画出训练损失曲线
        plt.plot(list(range(1,epoch+1)), loss_list, c="red")
        plt.xlabel(u'训练轮次', fontproperties='SimHei', color='red')
        plt.ylabel(u'loss', fontproperties='SimHei', color='red')
        plt.show()
    
    print(f'ok')