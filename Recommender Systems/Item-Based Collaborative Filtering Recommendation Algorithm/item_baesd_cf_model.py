# 基于物品的协同过滤算法

import math
# operator: 内置的操作符函数接口,它定义了一些算术和比较内置操作的函数
# itemgetter: 根据某个或某几个字典字段来排序列表
from operator import itemgetter

class ItemBasedCF(object):
    # 初始化参数
    def __init__(self, data_set):
        """
        :param data_set: 包含训练集和测试集的字典
        """
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.trainSet = data_set['trainSet']    # 索引训练集
        self.testSet = data_set['testSet']      # 索引测试集

        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    # 计算电影之间的相似度
    # 遍历训练数据，统计喜欢每个物品的用户数，存入movie_popular列表中，如movie_popular[i]表示喜欢电影i的用户数。
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():      # 循环取出一个用户和他看过的电影
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1      # 统计每部电影共被看过的次数
        self.movie_count = len(self.movie_popular)      # 得到电影总数
        print("Total movie number = %d" % self.movie_count)

        for user, movies in self.trainSet.items():      # 得到矩阵C，C[i][j]表示同时喜欢电影i和j的用户数
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    # self.movie_sim_matrix[m1][m2] += 1        # 同时喜欢电影m1和m2的用户+1    21.75  10.5   16.67
                    self.movie_sim_matrix[m1][m2] += 1 / math.log(1 + len(movies))  # ItemCF-IUF改进，惩罚了活跃用户 22.00 10.65 14.98
        print("Build co-rated users matrix success!")

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():        # 电影m1，及m1这行对应的电影们
            for m2, count in related_movies.items():        # 电影m2 及 同时看了m1和m2的用户数
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    # 计算出电影m1和m2的相似度
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')

        # 添加归一化    precisioin=0.2177	recall=0.1055	coverage=0.1497
        maxDict = {}
        max = 0
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, _ in related_movies.items():
                if self.movie_sim_matrix[m1][m2] > max:
                    max = self.movie_sim_matrix[m1][m2]

        for m1, related_movies in self.movie_sim_matrix.items():        # 归一化
            for m2, _ in related_movies.items():
            # self.movie_sim_matrix[m1][m2] = self.movie_sim_matrix[m1][m2] / maxDict[m2]
                self.movie_sim_matrix[m1][m2] = self.movie_sim_matrix[m1][m2] / max

    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = self.n_sim_movie        # 找到相似的20部电影
        N = self.n_rec_movie        # 为用户推荐10部
        rank = {}
        watched_movies = self.trainSet[user]        # 该用户看过的电影

        for movie, rating in watched_movies.items():        # 遍历用户看过的电影及对其评价
            # 找到与movie最相似的K部电影,遍历电影及与movie相似度
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:     # 如果用户已经看过了，不推荐了
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)        # 计算用户对该电影的兴趣
        # 返回用户最感兴趣的N部电影
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie        # 要推荐的电影数
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_moives = self.testSet.get(user, {})        # 测试集中用户喜欢的电影
            rec_movies = self.recommend(user)       # 得到推荐的电影及计算出的用户对它们的兴趣
            for movie, w in rec_movies:     # 遍历给user推荐的电影
                if movie in test_moives:    # 测试集中有该电影
                    hit += 1        # 推荐命中+1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_moives)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
