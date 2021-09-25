"""
@author: shencx
@time: 2021/6/28 0028 下午 7:32
@desc: 机器学习实战第五章-逻辑回归。代码基本一致，存在小的改动。注意，逻辑回归属于分类技术。
"""
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    data = []
    label = []
    fr = open('testSet.txt')
    for lines in fr.readlines():
        line = lines.strip().split()
        data.append([1.0, float(line[0]), float(line[1])])
        label.append(int(line[2]))
    fr.close()
    return data, label


def sigmoid(inx):
    if inx >= 0:
        return 1.0 / (1 + exp(-inx))
    else:
        return exp(inx) / (1 + exp(inx))      # 为了防止发生溢出（提示Runtimewarning）


def gradient(dataIn, labelIn):
    dataMatrix = mat(dataIn)
    labelMatrix = mat(labelIn).transpose()
    alpha = 0.001
    cycles = 500
    _, n = dataMatrix.shape
    weights = ones((n, 1))
    for cy in range(cycles):
        pre = sigmoid(dataMatrix * weights)
        error = labelMatrix - pre
        weights = weights + alpha * dataMatrix.transpose() * error  # 同上一行共同形成梯度更新的公式，公式推导利用了最大似然估计和梯度上升。
    return weights                                                  # 整体采用了批量法，权重的更新利用向量形式同步进行。


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    #dataMat = array(dataMat)
    n = array(dataMat).shape[0]
    xc1, yc1, xc2, yc2 = [], [], [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xc1.append(dataMat[i][1])                               # 可以用array形式，[][]或[,]来提取对应数值均可。
            yc1.append(dataMat[i][2])
        else:
            xc2.append(dataMat[i][1])
            yc2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xc1, yc1, c='y', s=20)
    ax.scatter(xc2, yc2, c='b', s=20)
    x = arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]        # 此处的y并不是输出，此行代码描绘的最佳分割线，是基于0=WX得到的。从下方所命名的横纵标题也可以看出
    y = y.transpose()
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def stroGradient(InData, InLabel):
    InData1 = array(InData)
    m, n = InData1.shape
    alpha = 0.01    # 学习率设为0.001时效果很差
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(InData1[i] * weights))
        error = InLabel[i] - h
        weights = weights + alpha * InData1[i] * error
    return weights


def stroGradientIm(Datain, Labelin, numInter=150):
    Datain1 = array(Datain)
    m, n = Datain1.shape
    weights = ones(n)
    for j in range(numInter):
        dataIndex = m
        for i in range(m):
            alpha = 4/(1 + j + i) + 0.01
            index = int(random.uniform(0, dataIndex))  # 原文使用了len(dataIndex),但对原文中以dataIndex=range(m)和del来实现的目的并不清楚。因为整个dataIndex只是用来生成随机数的上限。
            h = sigmoid(sum(Datain1[index] * weights))
            error = Labelin[index] - h
            weights = weights + alpha * error * Datain1[index]
            dataIndex -= 1
    return weights


# 马疝病数据集
def classify(inx, weights):
    pre = sigmoid(sum(inx * weights))
    if pre > 0.5:
        return 1.0
    else:
        return 0.0


def TestColic():
    FrTrain = open('horseColicTraining.txt')
    FrTest = open('horseColicTest.txt')
    # 进入训练阶段，获取最佳回归参数
    TrainSet = []
    TrainLabel = []
    for line in FrTrain.readlines():           # 这种方式不容易出错，但是似乎有更简便的方法。
        curLine = line.strip().split('\t')     # 这时已经是列表类型，内部应该是字符串格式。readlines返回的是列表，方便遍历。
        lineArr = []
        for i in range(len(curLine)):
            lineArr.append(float(curLine[i]))  # 不转还是字符串格式，无法进行计算。而且代码中多个变量以及数据本身都是浮点型，同时涉及混合计算，所以转成浮点型更合适。
        TrainSet.append(lineArr)
        TrainLabel.append(float(curLine[-1]))
    TrainWeight = stroGradientIm(TrainSet, TrainLabel, 500)
    # 进入测试阶段
    error_num = 0
    sam_num = 0
    for line in FrTest.readlines():
        sam_num += 1
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine)):
            lineArr.append(float(curLine[i]))
        pre_label = classify(lineArr, TrainWeight)
        if int(pre_label) != int(curLine[-1]):
            error_num += 1
    error_rate = float(error_num/sam_num)
    print('the error rate of the test dataset is {}'.format(error_rate))
    return error_rate


def MulTest():
    num_iter = 10
    error_sum = 0
    for i in range(num_iter):
        error_sum += TestColic()
    print('after {} iterations, the average error rate is {}'.format(num_iter, error_sum/float(num_iter)))  # error_sum是浮点型


if __name__ == '__main__':
    data, label = loadDataSet()  # 返回的是列表类型，后边计算都是mat矩阵或者是numpy数组
    # weight = stroGradient(data, label)
    weight = stroGradientIm(data, label, numInter=150)
    # weight = gradient(data, label)
    plotBestFit(weight)
    # MulTest()  # 测试结果与书内并不一致，未找到原因
