
import random

class DataPreProcess(object):
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if(random.random() < pivot):
                # 相当于trainSet.get(user)，若该键不存在，则设trainSet[user] = {}，典中典
                # 键中键：形如{'1': {'1287': '2.0', '1953': '4.0', '2105': '4.0'}, '2': {'10': '4.0', '62': '3.0'}}
                # 用户1看了id为1287的电影，打分2.0
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        data = {'trainSet': self.trainSet, 'testSet':self.testSet}
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)
        return data

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)
