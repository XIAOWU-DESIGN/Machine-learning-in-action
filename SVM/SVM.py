"""
@author: shencx
@time: 2021/7/27 0027 上午 9:02
@desc: 机器学习实战第六章支持向量机，公式推导借鉴的文章在代码注释中的链接
"""
from numpy import *


def load_data(filename):
    data = []
    label = []
    fr = open(filename)
    for line in fr.readlines():   # 函数读取所有行并以列表形式返回。
        line_arr = line.strip().split('\t')
        data.append([float(line_arr[0]), float(line_arr[1])])
        label.append(float(line_arr[2]))
    return data, label


def select_j(i, m):   # 仅在简单版的SMO使用，也就是第二个变量的选择是随机的。
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(alpha_j, h, l):   # 解出来的alpha是极值点，但是由于约束条件的限制，所以在保证alpha在范围内的同时尽可能接近极值点。
    if alpha_j > h:
        alpha_j = h
    if alpha_j < l:
        alpha_j = l
    return alpha_j


def smo_simple(data_in, labels, c, tolerance, max_iter):
    """
    @desc: 简化版的SMO，没有用核函数，第二个变量的选择是随机的。
    @author：shencx
    """
    data_mat = mat(data_in)
    label_mat = mat(labels).T
    b = 0
    m, n = shape(data_mat)
    alphas = mat(zeros((m, 1)))
    cur_iter = 0
    while cur_iter < max_iter:
        alphas_changed = 0
        for i in range(m):
            fx_i = float(multiply(alphas, label_mat).T * (data_mat * data_mat[i].T)) + b
            ei = fx_i - float(label_mat[i])
            if ((label_mat[i] * ei) < -tolerance and alphas[i] < c) or ((label_mat[i] * ei) > tolerance and alphas[i] > 0):  # 这里的判断条件可以试着将ei替换掉，就能发现了。其实就是违背KKT的条件。
                j = select_j(i, m)                                                                               # 整个判断句后的内容并不复杂，关键是要亲自推导过一次公式。我在网上也看了很多，虽然都写的不是多么严谨，但是够用。
                fx_j = float(multiply(alphas, label_mat).T * (data_mat * data_mat[j].T)) + b                     # 看过的、可以用的公式推导文章：https://blog.csdn.net/v_JULY_v/article/details/7624837  https://blog.csdn.net/on2way/article/details/47730367  https://blog.csdn.net/qq_30565883/article/details/100044805
                ej = fx_j - float(label_mat[j])
                alpha_i_old = alphas[i].copy()   # 深拷贝，主要是为了后边更新了alphas后再用到旧的alphas的第i和j个。
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:     # 根据alpha1 + alpha2 = alpha1_old + alpha2_old 求出的alpha2的范围。
                    lb = max(0, alphas[j] - alphas[i])   # 这几行用alpha_i_old和alpha_j_old代替更容易对上推导的公式，其实就是一样的。
                    hb = min(c, c + alphas[j] - alphas[i])
                else:
                    lb = max(0, alphas[j] + alphas[i] - c)
                    hb = min(c, alphas[j] + alphas[i])
                if lb == hb:
                    print("lb==hb")
                    continue
                eta = 2.0 * data_mat[i] * data_mat[j].T - data_mat[i] * data_mat[i].T - data_mat[j] * data_mat[j].T   # 这里和网站推导的多了个负号，所以后边alphas[j]的计算也略有不同。
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], hb, lb)   # 保证alphas[j]在约束范围内
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += label_mat[i] * label_mat[j] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i] * data_mat[i].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j] * data_mat[j].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i] * data_mat[i].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j] * data_mat[j].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphas_changed += 1
        if alphas_changed == 0:
            cur_iter += 1
        else:
            cur_iter = 0
        print("iteration number:", cur_iter)
    return b, alphas


# 完整版SMO
class OptStr:    # 创建一个数据结构来保存重要数值，方便了后续的传递和重复使用。使得代码写起来更简单点。
    def __init__(self, data_mat, labels, c, tolerance, kernel):
        self.x = data_mat
        self.label = labels
        self.c = c
        self.tol = tolerance
        self.m = shape(data_mat)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.cache = mat(zeros((self.m, 2)))   # 第一列存放cache是否有效的标志位，即对应的e是否计算好了。第二列就是存放计算出来的e
        self.K = mat(zeros((self.m, self.m)))  # 使用核函数时才能用到
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.x, self.x[i], kernel)  # 直接一开始就计算出所有样本之间的核函数结果。


def calculate_ek(ost, k1):
    # fx_k = float(multiply(ost.alphas, ost.label).T * (ost.x * ost.x[k].T) + ost.b)
    fx_k = float(multiply(ost.alphas, ost.label).T * ost.K[:, k1] + ost.b)  # 使用核函数时使用
    ek = fx_k - float(ost.label[k1])
    return ek


def sel_j(i, ost, ei):  # 寻找具有最大步长的第二个变量j，并返回此时的索引j和其对应的误差ej
    max_j = -1
    max_delta = 0
    ej = 0
    ost.cache[i] = [1, ei]
    cache_list = nonzero(ost.cache[:, 0].A)[0]   # 返回已经计算出e（标志位为1）的索引，并只取了行索引。nonzero返回非零或非False的元素的索引。
    if len(cache_list) > 1:
        for k1 in cache_list:
            if k1 == i:
                continue
            ek = calculate_ek(ost, k1)   # 按照cache_list的来源来讲，k对应的ek应该是已经计算出来的呀。
            delta_k = abs(ei - ek)
            if delta_k > max_delta:
                max_delta = delta_k
                max_j = k1
                ej = ek
        return max_j, ej
    else:          # 当第一次进入时，只有ei刚被添加进来，其他的都还未计算。所以随机选择一个j。
        j = select_j(i, ost.m)
        ej = calculate_ek(ost, j)
    return j, ej


def update_ek(ost, k1):
    ek = calculate_ek(ost, k1)
    ost.cache[k1] = [1, ek]


# 完整的Platt SMO的内循环(注意第一个变量i的选取并未说明，这是在外循环要解决的)
def inner(i, ost):
    ei = calculate_ek(ost, i)
    if (ost.label[i] * ei < -ost.tol and ost.alphas[i] < ost.c) or (ost.label[i] * ei > ost.tol and ost.alphas[i] > 0):
        j, ej = sel_j(i, ost, ei)
        alpha_i_old = ost.alphas[i].copy()
        alpha_j_old = ost.alphas[j].copy()
        if ost.label[i] != ost.label[j]:
            lb = max(0, ost.alphas[j] - ost.alphas[i])
            hb = min(ost.c, ost.c + ost.alphas[j] - ost.alphas[i])
        else:
            lb = max(0, ost.alphas[j] + ost.alphas[i] - ost.c)
            hb = min(ost.c, ost.alphas[j] + ost.alphas[i])
        if lb == hb:
            print("lb == hb")
            return 0          # 简易版中的continue替换成了return 0，代表alphas没有改变。
        # eta = 2.0 * ost.x[i] * ost.x[j].T - ost.x[i] * ost.x[i].T - ost.x[j] * ost.x[j].T
        eta = 2.0 * ost.K[i, j] - ost.K[i, i] - ost.K[j, j]  # 对应核函数的使用
        if eta >= 0:
            print("eta >= 0")
            return 0
        ost.alphas[j] -= ost.label[j] * (ei - ej) / eta
        ost.alphas[j] = clip_alpha(ost.alphas[j], hb, lb)
        update_ek(ost, j)   # 更改cache中对应位置的值
        if abs(ost.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0
        ost.alphas[i] += ost.label[j] * ost.label[i] * (alpha_j_old - ost.alphas[j])
        update_ek(ost, i)
        # b1 = ost.b - ei - ost.label[i] * (ost.alphas[i] - alpha_i_old) * ost.x[i] * ost.x[i].T - ost.label[j] * (
        #         ost.alphas[j] - alpha_j_old) * ost.x[i] * ost.x[j].T
        b1 = ost.b - ei - ost.label[i] * (ost.alphas[i] - alpha_i_old) * ost.K[i, i] - ost.label[j] * (ost.alphas[j] - alpha_j_old) * ost.K[i, j]
        # b2 = ost.b - ej - ost.label[i] * (ost.alphas[i] - alpha_i_old) * ost.x[i] * ost.x[i].T - ost.label[j] * (
        #         ost.alphas[j] - alpha_j_old) * ost.x[j] * ost.x[j].T
        b2 = ost.b - ej - ost.label[i] * (ost.alphas[i] - alpha_i_old) * ost.K[i, i] - ost.label[j] * (ost.alphas[j] - alpha_j_old) * ost.K[j, j]
        if 0 < ost.alphas[i] < ost.c:
            ost.b = b1
        elif 0 < ost.alphas[j] < ost.c:
            ost.b = b2
        else:
            ost.b = (b1 + b2) / 2.0
        return 1
    return 0


def smo_plus(data, labels, c, tolerance, max_iter, k_tup=('rbf', 1.3)):
    ost = OptStr(mat(data), mat(labels).transpose(), c, tolerance, kernel=k_tup)
    cur_iter = 0
    entire_set = True
    alpha_changed = 0
    while cur_iter < max_iter and alpha_changed > 0 or entire_set:
        alpha_changed = 0
        if entire_set:         # 两个循环分别对应两种选择第一个变量的方式（机器学习实战P99的下端）
            for i in range(ost.m):    # 遍历所有变量，作为第一个变量
                alpha_changed += inner(i, ost)
                print("full set, iter: {}, i: {}, pairs changed {}".format(cur_iter, i, alpha_changed))
            cur_iter += 1
        else:
            non_bound = nonzero((0 < ost.alphas.A) & (ost.alphas.A < c))[0]     # 每一个元素与0和c比较，各自返回True或False。试过0<alphas<c形式，但是不可以。
            for i in non_bound:   # 遍历选择不在边界0和c的变量作为第一个变量
                alpha_changed += inner(i, ost)
                print("non-bound, iter: {}, i: {}, pairs changed {}".format(cur_iter, i, alpha_changed))
            cur_iter += 1
        if entire_set:
            entire_set = False
        elif alpha_changed == 0:
            entire_set = True
        print("iteration number:", cur_iter)
    return ost.b, ost.alphas


def calculate_w(alphas, data, labels):  # 计算出y=wx+b中的w
    x1 = mat(data)
    label = mat(labels).transpose()
    m, n = shape(x1)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label[i], x1[i].T)
    return w


# 核函数部分
def kernel_trans(data, x1, k1):
    m, n = shape(data)
    ker = mat(zeros((m, 1)))
    if k1[0] == 'lin':
        ker = data * x1.T
    elif k1[0] == 'rbf':
        for i in range(m):
            delta = data[i] - x1
            ker[i] = delta * delta.T
        ker = exp(ker / (-1 * k1[1] ** 2))
    else:
        raise NameError('That kernel is not recognized')
    return ker


def test_rbf(k2=1.3):
    data_ar, label_ar = load_data('testSetRBF.txt')   # 实际使用来训练的数据集
    b, alpha = smo_plus(data_ar, label_ar, 200, 0.0001, 10000, k_tup=('rbf', k2))
    data_mat = mat(data_ar)
    label_mat = mat(label_ar).transpose()
    ind_support = nonzero(alpha.A > 0)[0]   # 找到所有支持向量。根据公式推导过程中，可以明白此做法的原因。
    vec_support = data_mat[ind_support]
    lab_support = label_mat[ind_support]
    print("there are {} support vector".format(shape(vec_support)[0]))
    m, n = shape(vec_support)
    error_count = 0
    for i in range(m):
        kernel_mat = kernel_trans(vec_support, data_mat[i], k1=('rbf', k2))  # 这里就明白为何前边只选取了支持向量，因为其他的样本对应的alpha为零，在计算w时没有任何作用。所以有没有都一样，不如就不去计算。
        predict = kernel_mat.T * multiply(lab_support, alpha[ind_support]) + b
        if sign(predict) != sign(label_ar[i]):  # sign是取符号的函数，数值大于零则返回1，小于零则返回-1
            error_count += 1
    print("the training error rate is {}".format(float(error_count) / m))
    # 使用全新的测试集对模型进行测试
    data_art, label_art = load_data('testSetRBF2.txt')
    error_count = 0
    data_matt = mat(data_art)
    label_matt = mat(label_art).transpose()
    m, n = shape(data_matt)
    for i in range(m):
        kernel_mat = kernel_trans(vec_support, data_matt[i], k1=('rbf', k2))   # 仍然是要用训练集的支持向量，因为整个模型是基于这些变量确定的w。
        predict = kernel_mat.T * multiply(lab_support, alpha[ind_support]) + b
        if sign(predict) != sign(label_matt[i]):
            error_count += 1
    print("the test error rate is {}".format(float(error_count) / m))


# 手写识别问题测试
def load_images(dir_name):
    from os import listdir
    hw_labels = []
    file_list_tr = listdir(dir_name)
    m = len(file_list_tr)
    tr_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = file_list_tr[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        if class_num == 9:   # 这次测试只做二分类，数字9和1的
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        tr_mat[i] = ima2vector('{}/{}'.format(dir_name, file_name_str))
    return tr_mat, hw_labels


def test_digits(k_t=('rbf', 10)):
    data_ar, label_ar = load_images('trainingDigits')  # 实际使用来训练的数据集
    b, alpha = smo_plus(data_ar, label_ar, 200, 0.0001, 10000, k_t)
    data_mat = mat(data_ar)
    label_mat = mat(label_ar).transpose()
    ind_support = nonzero(alpha.A > 0)[0]  # 找到所有支持向量。根据公式推导过程中，可以明白此做法的原因。
    vec_support = data_mat[ind_support]
    lab_support = label_mat[ind_support]
    print("there are {} support vector".format(shape(vec_support)[0]))
    m, n = shape(vec_support)
    error_count = 0
    for i in range(m):
        kernel_mat = kernel_trans(vec_support, data_mat[i], k_t)  # 这里就明白为何前边只选取了支持向量，因为其他的样本对应的alpha为零，在计算w时没有任何作用。所以有没有都一样，不如就不去计算。
        predict = kernel_mat.T * multiply(lab_support, alpha[ind_support]) + b
        if sign(predict) != sign(label_ar[i]):  # sign是取符号的函数，数值大于零则返回1，小于零则返回-1
            error_count += 1
    print("the training error rate is {}".format(float(error_count) / m))
    # 使用全新的测试集对模型进行测试
    data_art, label_art = load_images('testDigits')
    error_count = 0
    data_matt = mat(data_art)
    label_matt = mat(label_art).transpose()
    m, n = shape(data_matt)
    for i in range(m):
        kernel_mat = kernel_trans(vec_support, data_matt[i], k_t)  # 仍然是要用训练集的支持向量，因为整个模型是基于这些变量确定的w。
        predict = kernel_mat.T * multiply(lab_support, alpha[ind_support]) + b
        if sign(predict) != sign(label_matt[i]):
            error_count += 1
    print("the test error rate is {}".format(float(error_count) / m))


def ima2vector(filename):
    vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vector[0, 32 * i + j] = int(line[j])
    return vector


if __name__ == '__main__':
    # data_arr, label_arr = load_data('testSet.txt')
    # bb, alpha = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
    # print(bb)
    # print(alpha[alpha > 0])   # 呦呵，第一次见，这个东西还真挺好用
    # for i1 in range(100):
    #     if alpha[i1] > 0:   # 支持向量的定义，根据公式推导也知道了alpha为零的样本就是非支持向量。
    #         print(data_arr[i1], label_arr[i1])   # 书中的图6.4也很容易，只需要画散点图时把这几个支持向量用其他形式进行表示。

    # bb, alpha = smo_plus(data_arr, label_arr, 0.6, 0.001, 40, k_tup=('rbf', 1.3))   # 由于相关代码添加了核函数，若想得到与书籍相似结果，需要把核函数相关的部分去掉。
    # ws = calculate_w(alpha, data_arr, label_arr)
    # y = mat(data_arr)[2] * ws + bb
    # print(y, label_arr[2])

    # test_rbf()

    test_digits()   # 不得不说，效果这么好滴嘛！
