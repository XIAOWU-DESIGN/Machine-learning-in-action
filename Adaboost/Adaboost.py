"""
@author: shencx
@time: 2021/8/13 0013 下午 8:27
@desc: 机器学习第七章，集成学习的Adaboost，以决策数为基分类器
"""
from numpy import *


def simple_data():
    data_mat = mat([[1.0, 2.1],
                    [2.0, 1.1],
                    [1.3, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_mat, ind, thresh_value, thresh_ine):   # 依据某一个ind索引对应的特征进行决策，结果分别用1和-1来代表。并且分了lt（less than）和gt（great than）两种情况。
    pre_array = ones((shape(data_mat)[0], 1))
    if thresh_ine == 'lt':
        pre_array[data_mat[:, ind] >= thresh_value] = -1.0
    else:
        pre_array[data_mat[:, ind] < thresh_value] = -1.0
    return pre_array


def build_stump(data_arr, class_labels, d):   # d是一个概率分布向量，本质就是对每个数据样本赋予的权重。它的作用在后边再解释
    data_mat = mat(data_arr)
    label_mat = mat(class_labels).T
    m, n = shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class = mat(zeros((m, 1)))
    min_error = inf
    for i in range(n):   # 我们需要建立的是一个树桩（单层的决策树），需要遍历所有特征以找到一个最佳的划分属性以及划分值
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):   # 注意，在确定划分值时，并不是遍历每一个值，而是结合步长来遍历。用来确定阈值
            for ine in ['lt', 'gt']:   # 对于每一个根据步长大小确定的阈值，有两种决策方式（lt，gt）
                threshold = range_min + float(j) * step_size  # 放在这个for循环外更好
                predict_labels = stump_classify(data_mat, i, threshold, ine)  # 获取根据当前属性的划分值以及决策方式得到的决策结果
                error_mat = mat(ones((m, 1)))
                error_mat[predict_labels == label_mat] = 0  # 西瓜书P176公式8.18的一部分
                weight_error = d.T * error_mat   # 一个理想的基学习器的评价方法，可参考西瓜书P176公式8.18。选取此公式结果的最小的作为基学习器。
                if weight_error < min_error:   # 保留到目前为止最好的划分属性、划分值等内容
                    min_error = weight_error
                    best_class = predict_labels.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['ine'] = ine
    return best_stump, min_error, best_class


def adaboost_train(data_arr, class_labels, num_it=40):
    weak_class = []   # 用来保存每个基分类器的基本信息
    m = shape(data_arr)[0]
    d = mat(ones((m, 1)) / m)  # 对每个数据点的关注程度（权重），一开始是相等的。后边就会改变，会对分错的样本赋予更多的关注，总和是1是不变的，这是boosting族算法的工作机制。可参考西瓜书P173介绍
    agg_class = mat(zeros((m, 1)))
    for i in range(num_it):
        # print('d:{}'.format(d.T))
        best_stump, error, class_res = build_stump(data_arr, class_labels, d)  # 获得基分类器构建的的结果情况
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))   # 根据基分类器的错误率计算该分类器的权重alpha。参考西瓜书P175
        best_stump['alpha'] = alpha
        weak_class.append(best_stump)
        # print('class_res:{}'.format(class_res))
        exp_on = multiply(-1 * alpha * mat(class_labels).T, class_res)   # 接下来就是根据alpha去更新概率分布向量d了,以供应下一个基分类器的构建，公式推导参考西瓜书P175-176
        d = multiply(d, exp(exp_on))
        d = d / d.sum()
        agg_class += alpha * class_res  # 这里就是模型整体的输出结果。这个是Adaboost的加性模型解释下的推导，即Adaboost是基分类器的线性组合。可参考西瓜书P173
        # print('agg_class:{}'.format(agg_class.T))
        agg_errors = multiply(sign(agg_class) != mat(class_labels).T, ones((m, 1)))   # 注意对于Adaboost得到的结果要再用sign函数去获得最终的标签（因为经过了线性组合的计算）。这里是计算模型的错误情况。
        error_rate = agg_errors.sum() / m
        print('total error: {}'.format(error_rate))
        if error_rate == 0.0:   # 如果模型错误率已经降为0，则终止训练。注意，这个错误率不是针对基分类器的，而是基于基分类器构建出的Adaboost模型的，可以看到agg_class是累计产生的。
            break
    return weak_class


def adaboost_classify(data, classifier_arr):
    data_mat = mat(data)
    m = shape(data_mat)[0]
    agg_class = mat(zeros((m, 1)))
    for i in range(len(classifier_arr)):   # 按顺序遍历每一个基分类器，最终结果仍然就是基于基分类器的线性组合。
        class_pre = stump_classify(data_mat, classifier_arr[i]['dim'], classifier_arr[i]['threshold'], classifier_arr[i]['ine'])
        agg_class += classifier_arr[i]['alpha'] * class_pre
    return sign(agg_class)


def load_data(file_name):      # general function to parse tab -delimited floats
    num_feat = len(open(file_name).readline().split('\t'))    # get number of fields
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


if __name__ == '__main__':
    # data1, labels = simple_data()
    # classifier = adaboost_train(data1, labels, 30)
    # adaboost_classify([0, 0], classifier)

    data_arr1, label_arr1 = load_data('horseColicTraining2.txt')
    classifier1 = adaboost_train(data_arr1, label_arr1, 10)

    test_data, test_label = load_data('horseColicTest2.txt')
    predict_labels = adaboost_classify(test_data, classifier1)
    print(predict_labels)
    a = multiply(predict_labels != mat(test_label).T, ones((67, 1))).sum()   # 书中的方法我没能实现成功，于是模仿之前的一句代码来实现的。
    print(a)
