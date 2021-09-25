"""
@author: shencx
@time: 2021/7/19 0019 上午 9:09
@desc: 
"""
from numpy import *
import matplotlib.pyplot as plt


def load_data_set(filename):
    """
    @desc:加载数据集
    @author：shencx
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flo_line = list(map(float, cur_line))  #python2.x中map返回的是列表，所以在3.x中需要加上list。这里必须用map，不能直接float(curline),因为float只能处理string或者number形式。
        data_mat.append(flo_line)
    return data_mat


def dist(vector1, vector2):
    return sqrt(sum(power(vector1 - vector2, 2)))  # power(x, y)就是求x的y次方，而且可以x或y都可以是列表形式的。功能更强大。


def rand_cent(data_set, k):
    """
    @desc: 在数据集边界内，随机地计算出k个点，作为质心
    @author：shencx
    """
    n = data_set.shape[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * random.rand(k, 1)   # 注意，这里的加法是依靠广播机制进行的。另外，rand是以给定形式创建一个数组，数值为0到1之间的均匀分布的随机数。
    return centroids


def k_means(data_set, k):
    """
    @desc: k-means的实现
    @author：shencx
    """
    m = data_set.shape[0]
    centroids = rand_cent(data_set, k)
    cluster_ass = mat(zeros((m, 2)))  # 保留分配的结果情况
    cluster_change = True
    while cluster_change:
        cluster_change = False
        for i in range(m):
            min_dist = inf
            index = -1
            for j in range(k):    # 计算第i个样本与k个质心的距离，并判断其所属簇的标签。
                dist_j = dist(data_set[i], centroids[j])
                if dist_j < min_dist:
                    min_dist = dist_j
                    index = j
            if cluster_ass[i, 0] != index:   # 判断样本所属簇是否改变，直至所有样本都没有改变，才会保持False，然后终止while循环。
                cluster_change = True
            cluster_ass[i] = [int(index), float(min_dist**2)]
        # print(centroids)
        for cent1 in range(k):   # 对循环一次过后的各个簇，进行质点的计算更新。
            part_cluster = data_set[nonzero(cluster_ass[:, 0].A == cent1)[0]]   # 这一句首先是对clusterass转为数组，并与cent进行比较判断（这里应该也是用了广播机制吧？）得出True或False，
                                                                               # 而nonzero是返回数组中不为False和0的元素的索引（为两行的形式，第一行描述行，第二行描述列）
                                                                             # 注意，这里dataset的索引为一维数组形式，可以看出这样做也是可行的。
            centroids[cent1] = mean(part_cluster, axis=0)
    return centroids, cluster_ass


def bi_k_means(data_set, k, dist_means=dist):
    """
    @desc: 二分k-均值算法:克服kmeans收敛于局部最小值的问题。
    @author：shencx
    """
    m = data_set.shape[0]
    cluster_ass = mat(zeros((m, 2)))
    centroid0 = mat(mean(data_set, axis=0)).tolist()[0]
    cent_list = [centroid0]   # 用列表储存质心，并把第一个质心放入。
    for j in range(m):
        cluster_ass[j, 1] = dist_means(mat(centroid0), data_set[j])**2
    while len(cent_list) < k:
        lowest_sse = inf
        best_cent2split = -1
        best_new_cents = mat(zeros((2, data_set.shape[1])))
        for i in range(len(cent_list)):   # 此循环是对当前所有的已有簇进行逐个拆分并进行相关指标计算。最终选出最优的适合拆分的簇。
            part_cluster = data_set[nonzero(cluster_ass[:, 0].A == i)[0]]
            centroid_split, cluster_as_split = k_means(part_cluster, 2)   # 二分k-均值的体现
            sse_split = sum(cluster_as_split[:, 1])
            sse_not_split = sum(cluster_ass[nonzero(cluster_ass[:, 0].A != i)[0], 1])
            if (sse_split + sse_not_split) < lowest_sse:
                best_cent2split = i   # 记录最佳的被拆分的簇
                best_new_cents = centroid_split    # 记录新的两个质心
                best_cluster_ass = cluster_as_split.copy()   # 记录被拆分的簇的聚类结果
                lowest_sse = sse_split + sse_not_split
        # 对拆分的簇的聚类结果赋予标签，第一个相当于增加一个新簇标签（注意原来的标签最大到len(cent_list)-1，第二个相当于代替原来的簇标签。
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(cent_list)   # best_cluster_ass应该在for循环前进行定义的，但是它的形状无法确定，所以并未添加。
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent2split
        # 与上两行类似，我们需要将质心进行更新和增加新簇的质心。两个质心具体谁用来更新原来簇的质心都是无所谓的。
        cent_list[best_cent2split] = best_new_cents[0]
        cent_list.append(best_new_cents[1])
        # 将原有的第best_cent2split簇的所有样本的聚类情况更新为best_cluster_ass的内容。（找到该簇样本所有行，并赋新值）
        cluster_ass[nonzero(cluster_ass[:, 0].A == best_cent2split), :] = best_cluster_ass  # 不要将‘，’省略，否则会无法匹配。
    return cent_list, cluster_ass


# 由于剩余部分并未有太大价值，所以拷贝了源代码并进行修改。
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print(yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids1, clustAssing = bi_k_means(datMat, numClust, dist_means=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    myCentroids = zeros((numClust, 2))
    for i1 in range(3):
        myCentroids[i1] = array(myCentroids1[i1])
    ax1.scatter(myCentroids[:, 0], myCentroids[:, 1], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    # data_m = mat(load_data_set('testSet.txt'))
    data_m = mat(load_data_set('testSet2.txt'))
    # centroid = rand_cent(data_m, 2)
    # centroid, cluster_as = k_means(data_m, 4)
    centroid1, cluster_as = bi_k_means(data_m, 3)  # 一定要注意返回来的centroid1是列表（一维的），而其中的元素是matrix。
    centroid = zeros((3, 2))
    for i1 in range(3):
        centroid[i1, :] = array(centroid1[i1])
    # print(centroid)

    # 可以绘制图10-1,2,3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m1 = data_m.shape[0]
    marker = ['.', 'o', 'v', ',', '*']
    color = ['r', 'y', 'g', 'b', 'c']
    for ii in range(m1):
        s = int(cluster_as[ii, 0])
        ax.scatter(data_m[ii, 0], data_m[ii, 1], marker=marker[s], c=color[s])
    for jj in range(3):
        ax.scatter(centroid[jj, 0], centroid[jj, 1], marker=marker[-1], c=color[-1])
    plt.show()

    clusterClubs()
