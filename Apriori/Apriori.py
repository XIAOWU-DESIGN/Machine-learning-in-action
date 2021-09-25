"""
@author: shencx
@time: 2021/7/21 0021 上午 8:13
@desc: 机器学习实战第十一章，原理很简单，书中代码却让我头疼。实例部分未添加，烦了，躁了，不弄了，也不是多重要。
"""
from numpy import *


def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """
    @desc:以data_set中的所有单个元素构建大小为1的候选项集的集合
    @author：shencx
    """
    c1 = []
    for trans in data_set:
        for item in trans:
            if not [item] in c1:
                c1.append([item])
    c1.sort()   # 使得返回的以集合为元素的列表是按一定顺序排放的。
    # 将c1中的元素逐个做集合的映射。注意是每一个元素分别进行。
    return list(map(frozenset, c1))   # python3.x需要加上list，不然就只是一个迭代器，后边不能用。python2.x是返回列表。


def sup_can(data_set, ck, min_support):
    """
    @desc: 寻找频繁项集并记录其支持度
    @author：shencx
    """
    ss_cnt = {}   # 建立一个空字典以记录所有候选项集出现的次数。建立空集合要用set(),而如果是{1,2,3}则是建立的集合，因为其中没有‘键’
    for tid in data_set:
        for can in ck:   # 注意如果单独输出一个can是frozenset({x})的形式，但是使用时是不受影响的，这就类似于矩阵的matrix。
            # print(can)
            if can.issubset(tid):   # 针对集合的操作，判断can是否是tid的一个子集
                if can not in ss_cnt.keys():   # 判断can是否是ss_cnt的一个键
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_data = float(len(data_set))
    fre_list = []
    support_data = {}
    # print(ss_cnt)
    for key1 in ss_cnt:
        support = ss_cnt[key1]/num_data
        if support >= min_support:
            fre_list.insert(0, key1)  # 在列表前段插入key1，我觉得用append应该也一样。
            support_data[key1] = support  # 如果按照书中说的是频繁项集的支持度，则应该放在if里面。
    return fre_list, support_data


def apr_gen(l_k, k1):
    """
    @desc: 合成大小为k的集合，放于列表中并返回。
    @author：shencx
    """
    fre_list = []
    num_l = len(l_k)
    for i in range(num_l - 1):
        for j in range(i + 1, num_l):
            l_1 = list(l_k[i])[:k1-2].sort()   # 对这两行代码，书中所给的解释是有困惑的，网上的靠谱解释也很少，仅可提供：https://ask.csdn.net/questions/663985
            l_2 = list(l_k[j])[:k1-2].sort()   # 即使到代码写完也没明白，关联分析原理并不复杂，没想到实现起来让我这么意外的麻烦，哈哈哈哈
            if l_1 == l_2:                    # 其实网上也有很多其他人写的不同的实现方法
                fre_list.append(l_k[i] | l_k[j])
    return fre_list


def apriori(data_set, min_support=0.5):
    """
    @desc:完整的apriori算法。寻找到所有的频繁项集。
    @author：shencx
    """
    c1 = create_c1(data_set)
    data_set1 = list(map(set, data_set))
    print(data_set1)
    l_1, support_data = sup_can(data_set1, c1, min_support)
    fre_set = [l_1]   # 储存所有的频繁项集，先把大小为1的频繁项集放入。
    k2 = 2  # 除了大小为1的候选项集需要单独处理，其他的都有相似的方法，也就是apr_gen函数。首先就是合成大小为2的，所以k=2。
    while len(fre_set[-1]) > 0:   # 当刚生成的频繁项集为空时停止循环。完全不需要用k-2吧，至少目前没出错哈哈哈。
        c_k = apr_gen(fre_set[-1], k2)   # 生成最新的频繁项集的超集（大小为k）
        l_k, support_data_k = sup_can(data_set1, c_k, min_support)   # 寻找刚生成的超集中的频繁项集。
        # if len(l_k) == 0:   # 这样可以避免结果里面包含空的集合（while的条件直接用True），只是每次循环都要进行判断。
        #     break
        support_data.update(support_data_k)   # 将新频繁项集的支持度字典与已有的进行合并，相同的覆盖（本程序不涉及）。
        fre_set.append(l_k)
        k2 += 1
    return fre_set, support_data


def rules_gen(fre_data, sup_data, min_conf=0.7):
    """
    @desc: 注意！！！注意！！！是（前件+后件不变）前件缩减，后件（即h，h1）扩大的情况进行编写的。即根据一个规则不满足要求，则前件子集的规则一定不满足；前者满足，后者未必满足的逻辑。以后件为基础进行，有一个原因应该是集合的合并操作更容易且合理。
    @author：shencx
    """
    rule_list = []
    for i in range(1, len(fre_data)):   # 至少包含两个元素的频繁项集才能生成规则，fre_data[0]都是大小为1的频繁项集。
        for fre_seq in fre_data[i]:
            h1 = [frozenset([fre_d]) for fre_d in fre_seq]   # 将该频繁项集拆分成单个元素集合，以便后边进行剔除。
            if i > 1:
                h1 = calculate_conf(fre_seq, h1, sup_data, rule_list, min_conf)  # 应该是要有这个计算对单个元素的可信度，疑问了许久。最终找到了不错的讲解：https://blog.csdn.net/Gamer_gyt/article/details/51113753
                rules_seq(fre_seq, h1, sup_data, rule_list, min_conf)   #代码完全可以进一步合并改写，可见上述链接。
            else:
                calculate_conf(fre_seq, h1, sup_data, rule_list, min_conf)  # 大小为2的频繁项集可以直接进行可信度计算，不涉及元素合并。
    return rule_list


def calculate_conf(fre_seq, h, sup_data, brl, min_conf=0.7):
    """
    @desc: 计算任一频繁项集的所有规则的置信度（可信度），并保留有用的规则。
    @author：shencx
    """
    y_conf_list = []
    for con in h:
        conf = sup_data[fre_seq] / sup_data[fre_seq - con]
        if conf >= min_conf:
            print(fre_seq - con, '-->', con, 'conf:', conf)
            brl.append((fre_seq, con, conf))
            y_conf_list.append(con)
    return y_conf_list  # 注意返回的是后件


def rules_seq(fre_seq, h, sup_data, brl, min_conf=0.7):
    """
    @desc:　不断地合并集合形成新的前件，并进行可信度计算，直至判断条件不成立。
    @author：shencx
    """
    n = len(h[0])
    if len(fre_seq) > (n + 1):  #　在递归最后一步就会终止。
        hmp1 = apr_gen(h, n+1)
        hmp1 = calculate_conf(fre_seq, hmp1, sup_data, brl, min_conf)
        if len(hmp1) > 1:
            rules_seq(fre_seq, hmp1, sup_data, brl, min_conf)   # 递归，总感觉哪里不对


if __name__ == '__main__':
    # data = list(map(set, load_data()))   # 后续操作很多是基于集合形式的
    # c11 = create_c1(data)
    # print(c11)
    # l_11, sup_data1 = sup_can(data, c11, 0.5)
    # print(l_11, sup_data1)

    data = load_data()
    fre_set1, sup_data1 = apriori(data)
    rules = rules_gen(fre_set1, sup_data1, min_conf=0.7)
    print(rules)

    # 如果代码中还能看到这句话，就说明我不想弄机器学习实战书中的相关部分代码了。不想弄了!!!!!阿西吧。。一开始有理解错的地方，整完关联规则生成部分我就已经透透了。
