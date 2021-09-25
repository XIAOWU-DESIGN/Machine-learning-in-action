"""
@author: shencx
@time: 2021/8/6 0006 下午 3:23
@desc: 机器学习实战第九章，分类回归树CART，注意文中是以CART用于回归任务进行的，所以划分标准是采用的总方差而非基尼指数
"""

from numpy import *


def load_data(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flo_line = list(map(float, cur_line))
        data_mat.append(flo_line)
    return data_mat


def split_data(data_set, feature, value):
    mat0 = data_set[nonzero(data_set[:, feature] > value)[0]]   # 按照书中简单的测试也证明这里并不需要后边的那个[0]，书中应该是错了
    mat1 = data_set[nonzero(data_set[:, feature] <= value)[0]]
    return mat0, mat1


# 接下来是回归树的主体部分，要注意在做回归任务时，CART是以总方差(数据的离散程度)为判断依据，同时注意CART是二叉树而不是多叉树。
def reg_leaf(data_set):   # 不同于分类的叶节点有具体的类别标签（数值或字符），回归的结果本身并不固定(连续值，无穷个)，而这里就以均值作为结果（叶节点）。
    return mean(data_set[:, -1])


def reg_error(data_set):
    return var(data_set[:, -1]) * shape(data_set)[0]   # 正如书中所讲，这里是计算总方差，可以以var计算均方差再乘以样本个数。


# 用于计算出某节点处最佳划分的特征以及特征值
def choose_best_split(data_set, ops=(1, 4)):
    tol_s = ops[0]
    tol_n = ops[1]   # ops用元组传递了误差下降值的最小可容忍量，以及切分时允许的最小样本数，这两个值都对是否退出函数具有作用。
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:   # 所有结果值都一样，直接返回，该数据集不需要划分，
        return None, reg_leaf(data_set)
    m, n = shape(data_set)
    s_all = reg_error(data_set)
    s_best = inf
    index_best = 0
    value_best = 0
    for i in range(n-1):   # 最后一行是结果，所以总共n-1个特征
        for val in set((data_set[:, i].T.tolist())[0]):   # 一定要记住、学会set的巧妙用途。源代码有错，已修正。
            mat0, mat1 = split_data(data_set, i, val)
            if shape(mat0)[0] < tol_n or shape(mat1)[0] < tol_s:   # 用len函数也能返回行数，不过对于矩阵还是用shape稳点
                continue
            cur_s = reg_error(mat0) + reg_error(mat1)
            if cur_s < s_best:
                index_best = i
                value_best = val
                s_best = cur_s
    if s_all - s_best < tol_s:   # 当最佳划分仍然使得总反差没有下降程度超过容忍值，那就选择不划分
        return None, reg_leaf(data_set)
    # 隐掉的几行代码完全没用。前边for循环已经对所有情况（包括最佳划分）都进行了判断，这里再进行判断完全没有必要。
    # mat2, mat3 = split_data(data_set, index_best, value_best)
    # if shape(mat2)[0] < tol_n or shape(mat3)[0] < tol_s:
    #     return None, reg_leaf(data_set)
    return index_best, value_best   # 返回最佳划分特征的索引以及相应的划分特征值


def create_tree(data_set, ops=(1, 4)):
    feature, value = choose_best_split(data_set, ops)
    # feature, value = choose_best_split_model(data_set, ops)  # 模型树时选用
    if feature == None:  # 某分支节点处发现该节点没必要划分，则返回值value以形成叶节点
        return value
    tree = dict()
    tree['spInd'] = feature   # 相当于创建的节点。该节点的特征以及划分值
    tree['spVal'] = value
    left_set, right_set = split_data(data_set, feature, value)
    tree['left'] = create_tree(left_set, ops)
    tree['right'] = create_tree(right_set, ops)
    return tree


# 后剪枝。。。这一部分我有不是太确定的地方，所以如果注释有疑问，可以在网上进行查询。
def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 剪枝采用的应该是后剪枝中的错误率降低剪枝（REP）
def prune(tree, test_data):   # 代码包括书中都写的是测试集，准确来说是验证集。
    if shape(test_data)[0] == 0:     # 如果新的测试集没有划分到某节点任何数据，那么该节点领导的整个树被视为无用的，用均值形成叶节点代替，同时也避免了继续沿该分支往下迭代。这或许是在说明此分支是训练集过拟合造成的。
        return get_mean(tree)
    # 经过上述过程，断定该节点有用，则进行迭代部分。对于分支仍然是树的部分进行迭代
    if is_tree(tree['right']) or is_tree(tree['left']):   # 只要有一个分支仍然是树，就需要划分数据集以进行后续操作
        left_data, right_data = split_data(test_data, tree['spInd'], tree['spVal'])   # 获取该节点左右分支对应的测试数据集
    if is_tree(tree['left']):                           # 这两个判断句均是判定该分支仍然是树时，进行递归操作，直至找到分支下的叶节点。
        tree['left'] = prune(tree['left'], left_data)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right_data)
    # 这里相当于是迭代的一个出口，在分支均为叶节点时进行是否剪枝的判断。
    if not is_tree(tree['right']) and not is_tree(tree['left']):   # 经过上述迭代过程(未必执行)后左右分支都是叶节点，则计算不合并以及取平均值合并后的方差，比较后决定是否合并（剪枝）
        left_data, right_data = split_data(test_data, tree['spInd'], tree['spVal'])
        err_no_merge = sum(power(left_data[:, -1] - tree['left'], 2)) + sum(power(right_data[:, -1] - tree['right'], 2))  # 这时候是叶节点，里面是对应的回归预测值。所以利用power求误差（不合并时）
        mean_leaf = (tree['left'] + tree['right']) / 2.0
        err_merge = sum(power(test_data[:, -1] - mean_leaf, 2))   # 计算剪枝后的误差，注意用的数据
        if err_merge < err_no_merge:
            print('merging')
            return mean_leaf   # 对应剪枝误差更小，返回原左右叶节点的均值并以此替代树
        else:
            return tree    # 对应不剪枝误差更小，返回的树
    else:       #　经过上述迭代过程（一定被执行过）后仍然分支有树的情况，则将树返回。可以发现，当一个节点的任意一个分支是树时，该节点一定不需要考虑剪枝的问题。
        return tree


# 模型树部分。不同之处它的叶节点不是一个具体的数值，而是一个线性函数。整个树可以组成分段线性函数的结果。注意，这不是多变量决策树。
def linear_solve(data_set):
    m, n = shape(data_set)
    x = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    x[:, 1:n] = data_set[:, 0:n-1]
    y = data_set[:, -1]
    xtx = x.T * x
    if linalg.det(xtx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xtx.I * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    ws, x, y = linear_solve(data_set)
    return ws, x, y


def model_err(data_set):
    ws, x, y = linear_solve(data_set)
    y_pre = x * ws
    return sum(power(y - y_pre, 2))


def choose_best_split_model(data_set, ops=(1, 4)):   # 由于未采用书中的方式，这里需要另写一个函数。注意使用时替换create_tree中相应部分
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, model_leaf(data_set)
    m, n = shape(data_set)
    s_all = model_err(data_set)
    s_best = inf
    index_best = 0
    value_best = 0
    for i in range(n-1):
        for val in set((data_set[:, i].T.tolist())[0]):
            mat0, mat1 = split_data(data_set, i, val)
            if shape(mat0)[0] < tol_n or shape(mat1)[0] < tol_s:
                continue
            cur_s = model_err(mat0) + model_err(mat1)
            if cur_s < s_best:
                index_best = i
                value_best = val
                s_best = cur_s
    if s_all - s_best < tol_s:
        return None, model_leaf(data_set)
    return index_best, value_best


# 利用树进行预测
def reg_tree_eva(model):
    return float(model)


def model_tree_eva(model, in_data):
    n = shape(in_data)[1]
    x = mat(ones((1, n + 1)))
    x[:, 1:n+1] = in_data
    return float(x * model)


def tree_forecast(tree, in_data):   # 自上而下遍历整个树
    if not is_tree(tree):
        return reg_tree_eva(tree)
    if in_data[tree['spInd']] > tree['spVal']:   # 遍历时，在节点处进行测试数据对应特征与划分值的大小，从而判断进入左右分支
        if is_tree(tree['left']):   # 分支仍然是树，则迭代进行直至找到叶节点
            return tree_forecast(tree['left'], in_data)
        else:        # 分支为叶节点，则利用函数中的
            # return model_tree_eva(tree['left'], in_data)  # 使用模型树时采用
            return reg_tree_eva(tree['left'])
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], in_data)
        else:
            # return model_tree_eva(tree['right'], in_data)
            return reg_tree_eva(tree['right'])


def forecast(tree, test_data):
    m = len(test_data)
    y_pre = mat(zeros((m, 1)))
    for i in range(m):
        y_pre[i] = tree_forecast(tree, mat(test_data[i]))
    return y_pre


if __name__ == '__main__':
    # test_mat = mat(eye(4))   # 学到了eye新函数
    # mat00, mat11 = split_data(test_mat, 1, 0.5)
    # print(mat00, mat11)

    # test_mat = load_data('ex0.txt')
    # test_tree = create_tree(mat(test_mat))
    # print(test_tree)

    # my_data = mat(load_data('ex2.txt'))
    # my_tree = create_tree(my_data, ops=(0, 1))
    # print(my_tree)
    # test_data1 = mat(load_data('ex2test.txt'))
    # new_tree = prune(my_tree, test_data1)
    # print(new_tree)

    # data = mat(load_data('exp2.txt'))
    # tree1 = create_tree(data, ops=(1, 30))     # 很遗憾这个地方没能运行成功，出现了linear_solve函数中的错误提示。
    # print(tree1)

    # 模型树未能运行成功，所以这里仅做了回归树的测试
    train_data = mat(load_data('bikeSpeedVsIq_train.txt'))
    test_data = mat(load_data('bikeSpeedVsIq_test.txt'))
    reg_tree = create_tree(train_data, ops=(1, 20))
    y_pr = forecast(reg_tree, test_data[:, 0])
    print(corrcoef(y_pr, test_data[:, 1], rowvar=False))
