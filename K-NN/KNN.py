"""
@author: shencx
@time: 2021/7/13 0013 下午 7:09
@desc: 
"""
from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir


def create_data():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(x_test, x_train, labels, k):
    n = x_train.shape[0]
    diff_mat = tile(x_test, (n, 1)) - x_train   # 将x_test复制成（n,1）的形式，1是指一个x_test，并不是一个元素。这一步计算出了测试样本和所有训练样本的差。
    distances = ((diff_mat**2).sum(axis=1))**0.5   # 注意数组*计算是对应元素相乘，不是矩阵乘积。
    sort_dist = distances.argsort()             # 将元素从小到大排序，返回的是各元素对应的未排序时的索引值。
    class_count = {}   # 用字典来存放类别及其出现的次数
    for i in range(k):
        label = labels[sort_dist[i]]           # 获得索引值对应的类别
        class_count[label] = class_count.get(label, 0) + 1   # 从字典中获得label对应的值，当不存在该key值时取默认值0（并添加该key值）
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # items将字典以列表（内部为元组）形式返回；key决定了以何种依据进行排序（1就是指的第一个域）；True决定为降序
    return sort_class_count[0][0]   # 由上一行可以看出sort_class_count的形式


def file2matrix(filename):
    """
    @desc: 将数据转换成矩阵形式
    @author：shencx
    """
    fr = open(filename)
    array_lines = fr.readlines()    # 按行读取所有数据
    n = len(array_lines)
    f_mat = zeros((n, 3))
    label_vec = []
    index = 0
    for line in array_lines:
        list_line = line.strip().split('\t')
        f_mat[index] = list_line[:3]
        label_vec.append(int(list_line[-1]))   # 使用datingTestSet.txt数据集时，记得去掉int()
        index += 1
    return f_mat, label_vec


def norm(data_set):
    min_v = data_set.min(0)
    max_v = data_set.max(0)
    range_v = max_v - min_v
    n = data_set.shape[0]
    norm_data = data_set - tile(min_v, (n, 1))  # 注意tile的应用
    norm_data = norm_data / tile(range_v, (n, 1))  # 注意‘/’的计算方式，这里是对应元素相除。
    return norm_data, range_v, min_v


def dating_class_test():
    ratio = 0.1
    data_mat, data_label = file2matrix('datingTestSet.txt')
    norm_m, range_ss, min_val = norm(data_mat)
    n = norm_m.shape[0]
    num_test = int(n * ratio)
    print(num_test)
    error_count = 0
    for i in range(num_test):
        result = classify(norm_m[i], norm_m[num_test:], data_label[num_test:], 3)
        print('the classifier came back with: {}, the real answer is: {}.'.format(result, data_label[i]))
        if result != data_label[i]:
            error_count += 1
    print('the total error rate is: {}'.format(error_count / num_test))


def classify_person():    # 这段代码非常有趣，有交互。
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_ats = float(input('percentage of time spent playing video games?'))  # 注意float，很重要。
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    data_mat, data_label = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = norm(data_mat)
    in_arr = array([ff_miles, percent_ats, ice_cream])    # 注意这种将输入转换成可用数据的方式
    classify_result = classify((in_arr - min_val)/ranges, norm_mat, data_label, 3)   # 记得对inarr进行归一化
    print('you will probably like this person {}'.format(result_list[classify_result]))


# 手写识别系统
def ima2vector(filename):
    vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vector[0, 32 * i + j] = int(line[j])
    return vector


def hand_wri_test():
    tr_file_list = listdir('trainingDigits')  # 返回包含文件夹中所有文件/文件夹的名字的列表
    n = len(tr_file_list)
    tr_mat = zeros((n, 1024))
    tr_labels = []
    for i in range(n):
        file_name = tr_file_list[i]
        file_str = file_name.split('.')[0]   # 0代表保留‘.’之前的内容，而1代表保留其后的内容
        num_label = file_str.split('_')[0]   # 这两行保证获得该文件的名字的第一个数字（即文件内容对应的真实数字）
        tr_labels.append(num_label)
        tr_mat[i] = ima2vector('trainingDigits/{}'.format(file_name))  # 注意用的是filename
    # 进入测试阶段(test stage)
    te_file_list = listdir('testDigits')
    nte = len(te_file_list)
    error_count = 0
    for j in range(nte):
        file_name = te_file_list[j]
        file_str = file_name.split('.')[0]
        te_label = file_str.split('_')[0]
        te_mat = ima2vector('testDigits/{}'.format(file_name))
        cla_result = classify(te_mat, tr_mat, tr_labels, 3)   # 由于文件的内容由0,1组成，所以不需要进行归一化
        print('the classifier came back with: {}, the real answer is: {}.'.format(cla_result, te_label))
        if cla_result != te_label:
            error_count += 1.0
    print('the total error rate is: {}'.format(error_count/float(nte)))


if __name__ == '__main__':
    # x_tr, y = create_data()
    # label = classify([0.1, 0.1], x_tr, y, 3)
    # print(label)
    # matrix, label = file2matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(matrix[:, 0], matrix[:, 1], 15.0*array(label), 15.0*array(label))
    # plt.show()
    # norm_mat, ranges, min_val1 = norm(matrix)
    # print(norm_mat)
    # dating_class_test()
    # classify_person()
    hand_wri_test()
