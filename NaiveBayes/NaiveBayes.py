"""
@author: shencx
@time: 2021/8/10 0010 上午 10:01
@desc: 机器学习实战第4章，朴素贝叶斯分类器，使用了两种实现方式：基于伯努利模型、基于多项式模型。
"""
from numpy import *


# 这里针对文本中词是否出现作为的特征的方式被描述为词集模型
def load_data():
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]         # 留言文本的词条集合形式
    class_vec = [0, 1, 0, 1, 0, 1]   # 词条的类别标签，1代表是侮辱性文字，0代表正常言论
    return posting_list, class_vec


def vocab_list(data_set):   # 找到所有留言文本(data_set)中出现过的词汇，不重复。
    vocab_set = set([])
    for words in data_set:
        vocab_set = vocab_set | set(words)
    return list(vocab_set)


def words2vec(voc_list, input_words):     # 获取input_words对应的二元值向量化表示（反映其是否出现了对应的词汇）
    vec_input = [0] * len(voc_list)
    for word in input_words:
        if word in voc_list:
            vec_input[voc_list.index(word)] = 1    # 基于伯努利模型的实现方式，无论该词汇出现多少次，都记为1。也就是只需记录该词汇是否出现，用1和0来区分。二元值向量
        else:
            print('the word {} is not in vocabulary!'.format(word))
    return vec_input


def train_nb(train_mat, train_category):  # 训练集的作用就是通过具有标签的词条来确定每种标签下的词汇的出现概率，以供后续测试使用。
    num_doc = len(train_mat)    # 包含的词条（向量化表示）的个数
    num_words = len(train_mat[0])   # 每个词条的表示的长度
    p_cate = sum(train_category) / float(num_doc)   # 计算用于训练的词条属于侮辱性文字的概率。由于仅侮辱性的标签是1，所以可以求和来获得侮辱性词条的个数。
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_den = 0.0
    p1_den = 0.0
    for i in range(num_doc):
        if train_category[i] == 1:
            p1_num += train_mat[i]    # 将本词条中出现的所有词汇进行加1
            p1_den += sum(train_mat[i])   # 加该词条所出现的词汇的“数目”（其实并不是词条中出现的词汇真实个数，因为转成向量表示时，词汇的重复次数是体现不出来的）
        else:
            p0_num += train_mat[i]
            p0_den += sum(train_mat[i])
    p0_vec = log(p0_num / p0_den)   # 本质上就是计算出在正常言论中每种词汇出现的概率。注意是向量形式。log是为了防止下溢出，因为后续操作有太多很小的数需要相乘
    p1_vec = log(p1_num / p1_den)
    return p0_vec, p1_vec, p_cate


def classify_nb(vec_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_classify * p1_vec) + log(p_class1)    # 本来是计算条件概率的乘积，但是由于为了防止下溢出添加了log，所以就成了和的形式。
    p0 = sum(vec_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def te_nb():
    post_list, classes_list = load_data()  # 获得训练数据和标签
    voc_list = vocab_list(post_list)   # 找到训练数据的所有不重复词汇
    train_mat = []
    for post in post_list:
        train_mat.append(words2vec(voc_list, post))   # 将所有训练数据转成二元值向量，以便训练NB所用
    p0, p1, p_class1 = train_nb(array(train_mat), array(classes_list))   # 训练NB，获得两种类别下的词汇概率的向量化表示，以及类别1词条的出现概率

    test_entry = ['love', 'my', 'dalmation']
    test_p = array(words2vec(voc_list, test_entry))   # 获得测试词条对应的二元值向量
    print(test_entry, 'classified as', classify_nb(test_p, p0, p1, p_class1))
    test_entry = ['stupid', 'garbage']
    test_p = array(words2vec(voc_list, test_entry))   # 获得测试词条对应的二元值向量
    print(test_entry, 'classified as', classify_nb(test_p, p0, p1, p_class1))


# 过滤垃圾邮件的示例。这里我们提取特征换为词袋模型，也就是以每个词出现的次数作为特征
def words2vec_bag(voc_list, input_words):     # 获取input_words对应的向量化表示（反映词汇出现的次数）
    vec_input = [0] * len(voc_list)
    for word in input_words:
        if word in voc_list:
            vec_input[voc_list.index(word)] += 1    # 更改的位置，记录每个词汇出现的次数
        else:
            print('the word {} is not in vocabulary!'.format(word))
    return vec_input


def text_parse(big_string):
    import re
    list_tokens = re.split(r'\W', big_string)
    return [tok.lower() for tok in list_tokens if len(tok) > 2]   # 对文本进行了切分、筛选、小写


def spam_test():
    doc_list = []
    classes_list = []
    text_full = []
    for i in range(1, 26):   # 从两种类型的邮件文本中获得数据以及标签
        word_list = text_parse(open('email/spam/{}.txt'.format(i)).read())
        doc_list.append(word_list)
        text_full.extend(word_list)
        classes_list.append(1)
        word_list = text_parse(open('email/ham/{}.txt'.format(i)).read())   # 源文件的第23个有问题，其中的问号需要替换成空格。
        doc_list.append(word_list)
        text_full.extend(word_list)
        classes_list.append(0)

    voc_list = vocab_list(doc_list)  # 我对此处是有疑问的，这个列表作为训练集，理应去除其中被选作测试集的部分。所以我做了第123行的改动，但是我不确定是否正确
    train_index = list(range(50))   # 这个地方必须加上list，我不清楚2.x版本中不加list是否可以。
    test_index = []
    for i in range(10):   # 随机挑选出10个作为测试集（这里只保存了索引），并从训练集索引中剔除相应的值。
        rand_index = int(random.uniform(0, len(train_index)))
        test_index.append(train_index[rand_index])
        del(train_index[rand_index])
    train_mat = []
    train_class = []
    # voc_list = vocab_list([doc_list[i] for i in train_index])
    for doc_index in train_index:
        train_mat.append(words2vec_bag(voc_list, doc_list[doc_index]))   # 书中写错了，很明显这里他想用的是词袋模型的函数。
        train_class.append(classes_list[doc_index])
    p0, p1, p_class1 = train_nb(array(train_mat), array(train_class))
    error_count = 0
    for te_index in test_index:
        word_vec = words2vec_bag(voc_list, doc_list[te_index])   # 这里原文同样用错了函数
        if classify_nb(array(word_vec), p0, p1, p_class1) != classes_list[te_index]:
            error_count += 1
    print('the error rate is {}'.format(float(error_count / len(test_index))))


if __name__ == '__main__':
    # te_nb()
    spam_test()
