# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    # 花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度, 分类标签
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # 只选取前100条数据, 只有0,1两类, 0表示山鸢尾, 1表示变色鸢尾
    data = np.array(df.iloc[:100, [0,1,-1]])
    # 只选取前两列特征(花萼长度, 花萼宽度)
    return data[:,:2], data[:,-1]


class LRClassifier:
    def __init__(self, epochs=200, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        
    def linear(self, x):
        return np.dot(x, self.w)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # 增广特征向量
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # 增广权重向量
        self.w = np.zeros((X.shape[1], 1))
        # y增加一维
        y = np.expand_dims(y, axis=1)
        # 训练
        for epoch in range(self.epochs):
            z = self.linear(X)
            y_pred = self.sigmoid(z)
            error = y_pred - y
            self.w = self.w - self.lr * np.dot(X.T, error)
            print(f'loss: {np.abs(error.T.sum()) / X.shape[0]}')
            
    
if __name__ == '__main__':    
    
    # 数据准备和划分
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)    
    
    # 查看数据
    PLOT = False
    if PLOT:
        plt.scatter(X[:50,0],X[:50,1], label='0')
        plt.scatter(X[50:,0],X[50:,1], label='1')
        plt.xlabel(u'花萼长度(cm)', fontproperties='SimHei', color='red')
        plt.ylabel(u'花萼宽度(cm)', fontproperties='SimHei', color='red')
        plt.legend()
        plt.show()
    else:
        # 定义模型
        model = LRClassifier()
        # 训练
        model.fit(X_train, y_train)
        # 画出分类决策边界
        x_ponits = np.arange(4, 8)
        # 分类平面为 w1 * x1 + w2 * x2 + w0 = 0, 而图中是w2关于w1的函数, 转化为一条直线
        y_ = - (model.w[1] * x_ponits + model.w[0]) / model.w[2]
        plt.plot(x_ponits, y_, color='green')
        plt.scatter(X[:50,0],X[:50,1], label='0')
        plt.scatter(X[50:,0],X[50:,1], label='1')
        plt.xlabel(u'花萼长度(cm)', fontproperties='SimHei', color='red')
        plt.ylabel(u'花萼宽度(cm)', fontproperties='SimHei', color='red')
        plt.legend()
        plt.show()
        
        # todo 损失随着训练下降图像
        