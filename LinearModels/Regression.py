"""
@author: shencx
@time: 2021/6/29 0029 下午 7:31
@desc: 机器学习第八章线性回归
"""
from numpy import *
import matplotlib.pyplot as plt


def load_data(filename):
    num_fea = len(open(filename).readline().split('\t')) - 1  # 读取了第一行
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_fea):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


# 标准线性回归
def stand_regre(inx, iny):
    x_mat = mat(inx)
    y_mat = mat(iny)
    xtx = x_mat.T * x_mat
    if linalg.det(xtx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xtx.I * x_mat.T * y_mat.T    # y是否需要转置，是根据y=wx得到的y的真实维数，然后根据相应公式反推求w时y的维数（是否需要转置）。所以y的转置放置此处有利于理解。
    return ws


# 局部加权线性回归
def lwlr_model(cur_x, x_arr, y_arr, k=1):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    m = shape(x_mat)[0]
    weights = mat(eye(m))
    for j in range(m):
        res_x = cur_x - x_mat[j, :]
        weights[j, j] = exp(res_x * res_x.T/(-2.0 * k ** 2))
    xtx = x_mat.T * (weights * x_mat)
    if linalg.det(xtx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xtx.I * (x_mat.T * (weights * y_mat.T))        # 加括号虽然不影响结果，但是可以合理的减少计算量。可以自行搜索矩阵连乘的最佳加括号方式。
    return cur_x * ws   # 返回了该样本的预测值


def lwlr_test(xte, x_arr, y_arr, k=1):
    m = shape(xte)[0]
    y_pre = zeros(m)
    for i in range(m):
        y_pre[i] = lwlr_model(xte[i], x_arr, y_arr, k)
    return y_pre


def error(y_arr, y_pre):
    return ((y_arr - y_pre)**2).sum()


# 岭回归，一种重要的回归模型。采用了L2正则化。
def ridge_regress(x_mat, y_mat, lam=0.2):
    xtx = x_mat.T * x_mat
    den = xtx + eye(shape(x_mat)[1]) * lam
    if linalg.det(den) == 0:
        print('This matrix is singular, cannot be inverse')
        return
    ws = den.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    print(shape(y_mat))
    y_mean = mean(y_mat, 0)  # 对每一列求均值
    y_mat = y_mat - y_mean
    x_mean = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_mean)/x_var
    num_test = 30
    w_mat = zeros((num_test, shape(x_mat)[1]))
    for i in range(num_test):
        ws = ridge_regress(x_mat, y_mat, exp(i-10))
        w_mat[i] = ws.T
    return w_mat


def stage_wise(x_arr, y_arr, rate1=0.01, num_it=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_arr, 0)
    y_mat = y_mat - y_mean
    x_mean = mean(x_mat, 0)                # 此处开始的三行就是regularize函数
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_mean)/x_var
    _, n = shape(x_mat)
    return_ws = zeros((num_it, n))                 # 储存每次迭代后得到的所有权重
    ws = zeros((n, 1))
    ws_best = ws.copy()    # 删掉也仍能运行，但是会有提示“在赋值前可能会引用局部变量”
    for i in range(num_it):
        low_error = inf                        # inf为python中的无穷大表示
        for j in range(n):                    # 每一次仅针对一个特征，改变其对应的权重，并且进行权重增加或减少两种情况（即s=1，-1）
            for s in [-1, 1]:
                ws_test = ws.copy()      # 注意‘=’和np.copy的区别：前者的结果是所有变量都指向同一个地址。后者是深拷贝，赋值内容到新的地址，前后内容独立，修改时互不影响。
                                        # 所以可以看出，s=1，-1的两种结果都是基于同一个ws进行变化后得到的
                ws_test[j] += rate1 * s
                y_test = x_mat * ws_test          # 基于全部样本进行的权重测试
                re_error = error(y_test.A, y_mat.A)    # A是将矩阵转换为数组格式。作者猜测一个原因（未必准确）是：mat的**计算是两个矩阵相乘，而数组的**的计算是元素逐个平方。从error函数来看，应该选用后者
                if re_error < low_error:
                    low_error = re_error
                    ws_best = ws_test.copy()    # 本人认为这里应该也用copy，否则wstest的后续改变会影响wsbest
        ws = ws_best.copy()   # 注意此函数的多个变量实际是应该互相独立的，因为存在某个变量在多次循环中应保持不变的情况，如果不用copy，其值很容易被改变。
        return_ws[i] = ws.T
    return return_ws


# 剩余两部分对本人没有学习价值,并且很简单，拷贝了源代码并进行了修改，以补全内容。
from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print('%d\t%d\t%d\t%f\t%f' % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridge_test(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = error(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print('the best model from Ridge Regression is:\n', unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))


if __name__ == '__main__':
    # x, y = load_data('ex0.txt')   # 注意原始数据，第一列全为1，起到偏置项的作用；第二列为真正的输入；第三列为输出，即理想的回归值。注意y的维数
    # w = stand_regre(x, y)
    # x_Mat = mat(x)
    # y_Mat = mat(y)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], c='r')   # flatten是将矩阵进行一维降维，默认为逐行横向拼接，返回仍为矩阵。而A就是将矩阵转换成数组。
    # x_copy = xMat.copy()
    # x_copy.sort(0)
    # y_pre = x_copy * w
    # y_predict = lwlr_test(x, x, y, 0.003)
    # ind = xMat[:, 1].argsort(0)   # 返回数值从小到大的索引值
    # print(shape(xMat))
    # x_sort = xMat[ind][:, 0, :]
    # ax.plot(x_sort[:, 1], y_predict[ind])   # 画线一定要注意数据的顺序，不然画出的线有折返。
    # plt.show()
    # print(corrcoef(y_pre.T, y))

    # 岭回归
    x, y = load_data('abalone.txt')     # 注意y的shape为（1,4177）
    #weight = ridgetest(x, y)
    weight = stage_wise(x, y, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weight)
    plt.show()
