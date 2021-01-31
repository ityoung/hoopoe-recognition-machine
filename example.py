## （1）导入所需要的库
# -*- coding : utf-8 -*-
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
## （2）生成随机数据x及目标y
# 设置随机种子，生成同一份数据，方便多种方法比较
np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100,1)
y = 3*np.power(x,2) + 2 + 0.2*np.random.rand(x.size).reshape(100,1)
## (3)查看x,y数据分布情况
plt.scatter(x,y)
plt.show()
## (4)初始化权重参数
w1 = np.random.rand(1,1)
b1 = np.random.rand(1,1)
## （5）训练模型
lr =0.001  # 学习率
for i in range(800):
    # 前向传播
    y_pred = np.power(x,2)*w1+b1
    # 定义损失函数
    loss = 0.5 * (y_pred - y) **2
    loss = loss.sum
    # 计算梯度
    grad_w = np.sum((y_pred - y)*np.power(x,2))
    grad_b = np.sum(y_pred - y)
    # 使用梯度下降法，使得loss最小
    w1 -= lr * grad_w
    b1 -= lr * grad_b
## 可视化结果
plt.plot(x, y_pred, 'r-', label='predict')
plt.scatter(x,y,color='blue',marker='o',label='true')
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1,b1)