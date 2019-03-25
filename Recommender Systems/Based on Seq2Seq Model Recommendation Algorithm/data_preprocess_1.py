from __future__ import print_function

import random

class DataPreprocess1(object):
    def __init__(self, dataset_path):
        """
        :param dataset_path: 数据集文件路径
        """
        self.dataset_path = dataset_path
        # 将数据集划分为训练集和测试集
        self.train_set = {}
        self.test_set = {}

    def get_dataset(self, pivot=0.75):
        train_set_len = 0
        test_set_len = 0
        for line in self.load_dataset():
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:

    # 读文件，返回文件的每一行
    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0 :     # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % self.dataset_path)

    def load_data(self, filename, columns, separator):
        '''
        从文件名加载数据，并根据时间戳对其排序 Load the data from filename and sort it according to timestamp.
        返回一个包含3列的dataframe: user_id, item_id, rating Returns a dataframe with 3 columns: user_id, item_id, rating
        '''
        print('Load data...')

        data = pd.read_csv(filename, sep=separator, names=list(columns), index_col=False, usecols=range(len(columns)))

        if 'r' not in columns:
            # Add a column of default ratings
            data['r'] = 1

        if 't' in columns:
            # sort according to the timestamp column
            if data['t'].dtype == np.int64:  # probably a timestamp
                data['t'] = pd.to_datetime(data['t'], unit='s')
            else:
                data['t'] = pd.to_datetime(data['t'])
            print('Sort data in chronological order...')
            data.sort_values('t', inplace=True)
        return data