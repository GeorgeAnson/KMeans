# KMeans
Kmeans是一种无监督的基于距离的聚类算法，其变种还有Kmeans++。其中，sklearn中KMeans的默认使用的即为KMeans++。
KMeans
Kmeans是一种无监督的基于距离的聚类算法，其变种还有Kmeans++。其中，sklearn中KMeans的默认使用的即为KMeans++。

K-Means的工作原理
在K-Means算法中，簇的个数K是一个超参数，需要人为输入来确定。K-Means的核心任务就是根据设定好的K，找出K个最优的质心，并将离这些质心最近的数据分别分配到这些质心代表的簇中去。具体过程可以总结如下：
首先随机选取样本中的K个点作为聚类中心；
分别算出样本中其他样本距离这K个聚类中心的距离，并把这些样本分别作为自己最近的那个聚类中心的类别；
对上述分类完的样本再进行每个类别求平均值，求解出新的聚类质心；
与前一次计算得到的K个聚类质心比较，如果聚类质心发生变化，转过程b，否则转过程e；
当质心不发生变化时（当我们找到一个质心，在每次迭代中被分配到这个质心上的样本都是一致的，即每次新生成的簇都是一致的，所有的样本点都不会再从一个簇转移到另一个簇，质心就不会变化了），停止并输出聚类结果。

综上，K-Means 的算法步骤能够简单概括为：
1-分配：样本分配到簇。
2-移动：移动聚类中心到簇中样本的平均位置。

Kmeans优缺点
优点：

1.容易理解，聚类效果不错，虽然是局部最优， 但往往局部最优就够了；
2.处理大数据集的时候，该算法可以保证较好的伸缩性；
3.当簇近似高斯分布的时候，效果非常不错；
4.算法复杂度低。

缺点：

1.K 值需要人为设定，不同 K 值得到的结果不一样；
2.对初始的簇中心敏感，不同选取方式会得到不同结果；
3.对异常值敏感；
4.样本只能归为一类，不适合多分类任务；
5.不适合太离散的分类、样本类别不平衡的分类、非凸形状的分