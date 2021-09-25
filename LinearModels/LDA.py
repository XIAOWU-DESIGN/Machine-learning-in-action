"""
@author: shencx
@time: 2021/7/12 0012 下午 9:23
@desc: linear discriminant analysis (Fisher判别分析)，西瓜书公式推导。代码以二分类为例
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy import *


def loadData():
    data = datasets.load_iris()
    xTrain, xTest, yTrain, yTest = train_test_split(data.data, data.target, test_size=0.2, random_state=1)
    return xTrain, xTest, yTrain, yTest


def compute(x):
    """
    @desc: 计算一类样本的均值和方差
    @author：shencx
    """
    u = mean(x, 0)
    cov = zeros((array(x).shape[1], array(x).shape[1]))
    for s in x:
        re = s - u
        cov += re.reshape(2, 1) * re           # 注意类内散度矩阵的概念，注意西瓜书中的公式，得到的结果是矩阵形式。
    return cov, u


def fisher2(x1, x2):
    """
    @desc: 计算权重
    @author：shencx
    """
    cov1, u1 = compute(x1)
    cov2, u2 = compute(x2)
    s_w = cov1 + cov2
    u, s, v = linalg.svd(s_w)   # 西瓜书中有提及：实践中为了考虑到数值解的稳定性，通常对类间散度矩阵进行奇异值分解。
    s_w_1 = dot(dot(v.T, linalg.inv(diag(s))), u.T)   # 求s_w的逆。注意diag的运用，形成对角矩阵。
    w = dot(s_w_1, (u1 - u2))
    return w


def classification(sample, w, x1, x2):
    """
    @desc: 对sample的类别进行分类
    @author：shencx
    """
    u1 = mean(x1, 0)
    u2 = mean(x2, 0)
    print(u2)
    center1 = dot(w.T, u1)
    center2 = dot(w.T, u2)
    point = dot(w.T, sample)
    if abs(point - center1) < abs(point - center2):         # 根据投影点到两类样本的中心点投影点的距离大小来判断类别
        label = 1                  # 在所参考的网址中，作者提到了一个阈值的计算以及利用阈值来判断类别，但是在其代码中似乎并没有采用阈值。
    else:
        label = 2
    return label


if __name__ == '__main__':
    # 代码参考:https://blog.csdn.net/abc13526222160/article/details/90611743
    x, y = datasets.make_multilabel_classification(n_samples=20, n_features=2,
                                          n_labels=1, n_classes=1,
                                          random_state=2)  # 设置随机数种子，保证每次产生相同的数据。
    # 根据类别分个类
    index1 = array([index for (index, value) in enumerate(y) if value == 0])  # 获取类别1的indexs  先进行for  in  再进行if， 为真则放入数组
    index2 = array([index for (index, value) in enumerate(y) if value == 1])  # 获取类别2的indexs
    x_1 = x[index1]  # 类别1的所有数据(x1, x2) in X_1
    x_2 = x[index2]  # 类别2的所有数据(x1, x2) in X_2

    w = fisher2(x_1, x_2)
    label = classification(x_1[3], w, x_1, x_2)
    print(label)
    # 多分类也并不复杂，只要公式能推导即可。代码实现差不太多。
