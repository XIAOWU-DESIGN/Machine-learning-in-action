"""
@author: shencx
@time: 2021/8/1 0001 下午 3:35
@desc: 机器学习第三章决策树。ID3构造算法
"""
from numpy import *
from math import log
import matplotlib.pyplot as plt


def calculate_ent(data_set):
    n = len(data_set)
    label_count = {}
    for feat_vec in data_set:  # 利用字典去记录每种类别的数据的数目
        label = feat_vec[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    ent = 0.0
    for k in label_count:     # 计算信息熵ent
        prob = float(label_count[k]) / n
        ent -= prob * log(prob, 2)
    return ent


def create_data():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']   # 特征（属性）的名称。对应了data_set前两列。
    return data_set, labels


def split_data(data_set, axis, value):
    re_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            re_feat_vec = feat_vec[:axis]    # 注意划分后的数据不包含用于划分的特征！！！
            re_feat_vec.extend(feat_vec[axis+1:])   # 注意extend的用法。直接连在之前的列表后，处于同一个列表。
            re_data_set.append(re_feat_vec)
    return re_data_set


def choose_best_feature2split(data_set):  # 决策树的核心环节。解决当前分支结点处如何选取最佳的划分属性（特征）的问题。
    n_features = len(data_set[0]) - 1
    ent_all = calculate_ent(data_set)
    best_gain = 0.0
    best_feature = -1
    for i in range(n_features):
        feat_list = [feat_vec[i] for feat_vec in data_set]
        feat_value = set(feat_list)   # 注意set创建时是无序不重复的。以上两行便可以获取第i个特征的所有可能取值。这一行很巧妙也很关键。
        ent_sub = 0.0
        for val in feat_value:   # 计算该特征下的所有可能取值的子数据集以及各自的信息熵，并更新该特征的信息熵
            sub_data = split_data(data_set, i, val)
            prob = len(sub_data) / len(data_set)
            ent_sub += prob * calculate_ent(sub_data)
        sub_gain = ent_all - ent_sub
        if sub_gain > best_gain:   # 判断得到最佳的划分属性
            best_gain = sub_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # items将字典以列表（内部为元组）形式返回；key决定了以何种依据进行排序（1就是指的第一个域）；True决定为降序
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    # 迭代进行，所以先考虑迭代终止条件。根据实际情况这里要放在开端。
    class_list = [vec[-1] for vec in data_set]
    if class_list.count(class_list[0]) == len(class_list):   # 第一个终止条件：当前所有数据的类别都是一样的。（与第一个元素相同的元素数目等于总的元素数目）
        return class_list[0]
    if len(data_set[0]) == 1:   # 数据集已经不包含特征数据了（仅包含最后一列的类别标签了），也就表明所有特征都被遍历过了。
        return majority_cnt(class_list)
    # 获取最佳划分属性并将其作为节点保存于字典形式的树中
    best_feature = choose_best_feature2split(data_set)  # 返回的是特征的索引
    best_label = labels[best_feature]
    my_tree = {best_label: {}}    # 这个地方可以借助最终的输出形式来理解这个树究竟是如何一步步保存的。递归的结果是一步步返回而形成的。最终输出：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    # 按照最佳划分属性将数据集划分开，并且对划分后的数据集重复上述操作
    del(labels[best_feature])  # 删掉已经成为最佳划分属性的元素
    feature_values = [vec1[best_feature] for vec1 in data_set]
    unique_value = set(feature_values)
    for val in unique_value:
        sub_labels = labels.copy()   # 因为要不断地循环，而每次循环都要用同样的去除了之前最佳划分属性的labels，同时，传入的参数是列表类型，所以为了避免原始labels被修改，需要用新的来代替。但是我不明白为什么用等于号没有出错。为了更安全，我选用了copy。
        my_tree[best_label][val] = create_tree(split_data(data_set, best_feature, val), sub_labels)  # 递归嵌套。注意my_tree[best_label]仍然代表一个字典，这里添加了val键值。这个地方需要对基本的决策树实现流程有足够的了解。接下来的流程是基于以最佳划分属性取不同值所划分的数据集（不包含最佳划分属性）进行的。
    return my_tree


# 决策树的应用
def classify(tree_model, feat_labels, test_vec):
    first_str = list(tree_model.keys())[0]
    second_dict = tree_model[first_str]
    feat_index = feat_labels.index(first_str)   # 返回所查找对象在列表中的索引值。确定出该节点所划分的属性，并且在后边的test_vec中找到相对应的属性的值。这里很关键。
    class_label = ''
    for k in second_dict.keys():
        if test_vec[feat_index] == k:   # k有多个取值，在这数次的循环中仅会有一次成立。这时才进入下一个判断节点或者是叶节点，得到最终的类别标签。
            if type(second_dict[k]).__name__ == 'dict':  # 就相当于判断该节点是否是判断节点，因为只有叶节点才是一个非字典类型。
                class_label = classify(second_dict[k], feat_labels, test_vec)   # 仍然是判断节点，则需要继续进行迭代。就像是在剥洋葱。从前边开头几行的代码可以看出代码的运作规律，所以这里要往后传递的不是整个的决策树，而是以该节点为首的部分。
            else:
                class_label = second_dict[k]   # 说明是叶节点，则对应的值就是类别标签了。
    return class_label


def store_tree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# 决策树的图形表示。  这一部分我用的源代码，仅进行了整理并未运行。
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

# if you do get a dictonary you know it's a tree, and the first element will be another dict
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


if __name__ == '__main__':
    data, lab = create_data()
    label1 = lab.copy()  # 一开始我没有这一步，也就是后边这两个函数都用了同一个变量lab，导致第二个函数的输入lab其实是被改变了的，因为传入的是列表，是可以被改变的。这就是我在create_tree函数中注释说的问题。
    tree = create_tree(data, label1)
    pre = classify(tree, lab, [1, 0])
    print(pre)
