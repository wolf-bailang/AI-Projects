from __future__ import print_function

import numpy as np
import pandas as pd
# import random
# import argparse
import os
import sys
from shutil import copyfile

class DataPreprocess(object):
    def __init__(self, filename):
        self.filename = filename
        self.columns = "uit"              #['u', 'i', 't']
        self.separator = "\s+"              # 列之间的分隔符
        self.min_user_activity = 2          # 交互少于此值的用户将从数据集中删除
        self.min_item_popularity = 5       # 交互少于此值的项将从数据集中删除
        self.val_size = 0.1         # 输入验证集的用户数量
        self.test_size = 0.1       # 输入测试集的用户数量

        self.dirname = os.path.dirname(os.path.abspath(self.filename)) + "/"

    # 询问用户是否确定要在该目录中创建文件。
    # Ask user if he's sure to create files in that directory.
    def warn_user(self):
        print('This program will create a lot of files and directories in ' + self.dirname)
        answer = input('Are you sure that you want to do that ? [y/n]')
        if answer != "y":
            sys.exit(0)

    def create_dirs(self):
        if not os.path.exists(self.dirname + "data"):
            os.makedirs(self.dirname + "data")
        if not os.path.exists(self.dirname + "models"):
            os.makedirs(self.dirname + "models")
        if not os.path.exists(self.dirname + "results"):
            os.makedirs(self.dirname + "results")

    # 从文件名加载数据，并根据时间戳对其排序。
    # Load the data from filename and sort it according to timestamp.
    def load_data(self):
        """
        :param columns:
        :param separator:
        :return: 返回一个包含3列的dataframe: user_id, item_id, rating。
        Returns a dataframe with 3 columns: user_id, item_id, rating
        """
        print('Load data...')
        data = pd.read_csv(self.filename, sep=self.separator, names=list(self.columns), index_col=False, usecols=range(len(self.columns)), error_bad_lines=False)
        if 'r' not in self.columns:
            # 添加一列默认评级ratings。Add a column of default ratings
            data['r'] = 1
        if 't' in self.columns:
            # 根据时间戳列进行排序。sort according to the timestamp column
            if data['t'].dtype == np.int64:  # 可能一个时间戳。probably a timestamp
                data['t'] = pd.to_datetime(data['t'], unit='s')
            else:
                data['t'] = pd.to_datetime(data['t'])
            print('Sort data in chronological order...')
            data.sort_values('t', inplace=True)
        return data

    # 删除出现在太少交互中的用户和项。
    # Removes user and items that appears in too few interactions.
    def remove_rare_elements(self, data):
        """
        :param data:
        :param min_user_activity: 是用户应该具有的最小交互次数。the minimum number of interaction that a user should have.
        :param min_item_popularity: 是一个项目应该具有的最小交互数量。the minimum number of interaction that an item should have.
        :return:
        NB: 项目上的约束可能不会得到严格满足，因为罕见的用户和项目是交替删除的,最后一次移除不活跃用户可能会创建新的稀有物品.
        the constraint on item might not be strictly satisfied because rare users and items are removed in alternance,
        and the last removal of inactive users might create new rare items.
        """
        print('Remove inactive users and rare items...')
        # 第一次删除不活动的用户.Remove inactive users a first time
        user_activity = data.groupby('u', as_index=False).size()
        data = data[np.in1d(data.u, user_activity[user_activity >= self.min_user_activity].index)]
        # 删除不太受欢迎的产品.Remove unpopular items
        item_popularity = data.groupby('i', as_index=False).size()
        data = data[np.in1d(data.i, item_popularity[item_popularity >= self.min_item_popularity].index)]
        # 删除由于删除稀有项而可能超过活动阈值的用户
        # Remove users that might have passed below the activity threshold due to the removal of rare items
        user_activity = data.groupby('u', as_index=False).size()
        data = data[np.in1d(data.u, user_activity[user_activity >= self.min_user_activity].index)]

        return data

    # 在dirname中将原始用户id和项id映射为连续的数字id。
    # Save the mapping of original user and item ids to numerical consecutive ids in dirname.
    def save_index_mapping(self, data):
        """
        :param data:
        :param separator:
        :param dirname:
        :return:
        NB:在前面的步骤中，可能已经删除了一些用户和项，因此不会出现在映射中
        some users and items might have been removed in previous steps and will therefore not appear in the mapping.
        """
        self.separator = "\t"

        # panda分类类型将创建我们想要的数字id
        # Pandas categorical type will create the numerical ids we want
        print('Map original users and items ids to consecutive numerical ids...')
        data['u_original'] = data['u'].astype('category')
        data['i_original'] = data['i'].astype('category')
        data['u'] = data['u_original'].cat.codes
        data['i'] = data['i_original'].cat.codes

        print('Save ids mapping to file...')
        user_mapping = pd.DataFrame({'original_id': data['u_original'], 'new_id': data['u']})
        user_mapping.sort_values('original_id', inplace=True)
        user_mapping.drop_duplicates(subset='original_id', inplace=True)
        user_mapping.to_csv(self.dirname + "data/user_id_mapping", sep=self.separator, index=False)

        item_mapping = pd.DataFrame({'original_id': data['i_original'], 'new_id': data['i']})
        item_mapping.sort_values('original_id', inplace=True)
        item_mapping.drop_duplicates(subset='original_id', inplace=True)
        item_mapping.to_csv(self.dirname + "data/item_id_mapping", sep=self.separator, index=False)

        return data

    # 将数据集分为训练集、验证集和测试集。
    # Splits the data set into training, validation and test sets.
    # 每个用户都在且仅在一个集合中。
    # Each user is in one and only one set.
    def split_data(self, data):
        """
        :param data:
        :param nb_val_users:是要放入验证集的用户数量。the number of users to put in the validation set.
        :param nb_test_users:是要放入测试集的用户数量。the number of users to put in the test set.
        :param dirname:
        :return:
        """
        nb_val_users = self.val_size
        nb_test_users = self.test_size

        nb_users = data['u'].nunique()
        # 检查nb_val_user是否指定为分数
        # check if nb_val_user is specified as a fraction
        if nb_val_users < 1:
            nb_val_users = round(nb_val_users * nb_users)
        if nb_test_users < 1:
            nb_test_users = round(nb_test_users * nb_users)
        nb_test_users = int(nb_test_users)
        nb_val_users = int(nb_val_users)

        if nb_users <= nb_val_users + nb_test_users:
            raise ValueError('Not enough users in the dataset: choose less users for validation and test splits')

        def extract_n_users(df, n):
            users_ids = np.random.choice(df['u'].unique(), n)
            n_set = df[df['u'].isin(users_ids)]
            remain_set = df.drop(n_set.index)
            return n_set, remain_set

        print('Split data into training, validation and test sets...')
        test_set, tmp_set = extract_n_users(data, nb_test_users)
        val_set, train_set = extract_n_users(tmp_set, nb_val_users)

        print('Save training, validation and test sets in the triplets format...')
        train_set.to_csv(self.dirname + "data/train_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False,
                         header=False)
        val_set.to_csv(self.dirname + "data/val_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False, header=False)
        test_set.to_csv(self.dirname + "data/test_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False,
                        header=False)

        return train_set, val_set, test_set

    # 从数据生成用户操作序列。
    # Generates sequences of user actions from data.
    def gen_sequences(self, data, half=False):
        '''
        每个序列的格式为[user_id, first_item_id, first_item_rating, 2nd_item_id, 2nd_item_rating，…]。
        each sequence has the format [user_id, first_item_id, first_item_rating, 2nd_item_id, 2nd_item_rating, ...].
        如果一半为真，则将序列的长度减半(这对于生成扩展训练集很有用)。
        If half is True, cut the sequences to half their true length (useful to produce the extended training set).
        '''
        # 归并排序是稳定的，并保持时间顺序
        # Mergesort is stable and keeps the time ordering
        data = data.sort_values('u', kind="mergesort")
        seq = []
        prev_id = -1
        for u, i, r in zip(data['u'], data['i'], data['r']):
            if u != prev_id:
                if len(seq) > 3:
                    if half:
                        seq = seq[:1 + 2 * int((len(seq) - 1) / 4)]
                    yield seq
                prev_id = u
                seq = [u]
            seq.extend([i, r])
        if half:
            seq = seq[:1 + 2 * int((len(seq) - 1) / 4)]
        yield seq

    # 以序列格式转换训练/验证/测试集并保存它们。
    # Convert the train/validation/test sets in the sequence format and save them.
    def make_sequence_format(self, train_set, val_set, test_set):
        '''
        还要创建扩展的训练序列，它包含验证和测试集中用户序列的前一半。
        Also create the extended training sequences, which countains the first half of the sequences of users in the validation and test sets.
        '''
        print('Save the training set in the sequences format...')
        with open(self.dirname + "data/train_set_sequences", "w") as f:
            for s in gen_sequences(train_set):
                f.write(' '.join(map(str, s)) + "\n")

        print('Save the validation set in the sequences format...')
        with open(self.dirname + "data/val_set_sequences", "w") as f:
            for s in gen_sequences(val_set):
                f.write(' '.join(map(str, s)) + "\n")

        print('Save the test set in the sequences format...')
        with open(self.dirname + "data/test_set_sequences", "w") as f:
            for s in gen_sequences(test_set):
                f.write(' '.join(map(str, s)) + "\n")

        # 序列+包含train_set_sequence的所有序列加上val和测试集的一半序列
        # sequences+ contains all the sequences of train_set_sequences plus half the sequences of val and test sets
        print('Save the extended training set in the sequences format...')
        copyfile(self.dirname + "data/train_set_sequences", self.dirname + "data/train_set_sequences+")
        with open(self.dirname + "data/train_set_sequences+", "a") as f:
            for s in gen_sequences(val_set, half=True):
                f.write(' '.join(map(str, s)) + "\n")
            for s in gen_sequences(test_set, half=True):
                f.write(' '.join(map(str, s)) + "\n")

    def save_data_stats(self, data, train_set, val_set, test_set):
        print('Save stats...')

        def _get_stats(df):
            return "\t".join(
                map(str, [df['u'].nunique(), df['i'].nunique(), len(df.index), df.groupby('u').size().max()]))

        with open(self.dirname + "data/stats", "w") as f:
            f.write("set\tn_users\tn_items\tn_interactions\tlongest_sequence\n")
            f.write("Full\t" + _get_stats(data) + "\n")
            f.write("Train\t" + _get_stats(train_set) + "\n")
            f.write("Val\t" + _get_stats(val_set) + "\n")
            f.write("Test\t" + _get_stats(test_set) + "\n")

    def make_readme(self, val_set, test_set):
        data_readme = '''
    	以下文件是由preprocess.py自动生成的
    	The following files were automatically generated by preprocess.py

    	user_id_mapping
    		原始数据集中的用户id与新用户id之间的映射。
    		mapping between the users ids in the original dataset and the new users ids.
    		第一列包含新id，第二列包含原始id。
    		the first column contains the new id and the second the original id.
    		非活动用户可能已经从原始用户中删除，因此它们不会出现在id映射中。
    		Inactive users might have been deleted from the original, and they will therefore not appear in the id mapping.

    	item_id_mapping
    		项目id的Idem。
    		Idem for item ids.

    	train_set_triplets
    		训练以三元组的形式进行。
    		Training set in the triplets format.
    		每一行都是表单中的用户项交互(user_id, item_id, rating)。
    		Each line is a user item interaction in the form (user_id, item_id, rating). 
    		交互作用按时间顺序列出。
    		Interactions are listed in chronological order.

    	train_set_sequences
    		训练设置为序列格式。
    		Training set in the sequence format.
    		每一行都包含表单中用户的所有交互(user_id, first_item_id, first_rating, 2nd_item_id, 2nd_rating，…)。
    		Each line contains all the interactions of a user in the form (user_id, first_item_id, first_rating, 2nd_item_id, 2nd_rating, ...).

    	train_set_sequences+
    		序列格式的扩展训练集。
    		Extended training set in the sequence format.
    		扩展训练集包含所有训练集，以及验证和测试集中每个用户交互的前一半。
    		The extended training set contains all the training set plus the first half of the interactions of each users in the validation and testing set.

    	val_set_triplets
    		验证设置为三元组格式
    		Validation set in the triplets format

    	val_set_triplets
    		验证设置为序列格式
    		Validation set in the sequence format

    	test_set_triplets
    		测试集采用三联格式
    		Test set in the triplets format

    	test_set_triplets
    		序列格式的测试集
    		Test set in the sequence format

    	stats
    		包含有关数据集的一些信息。
    		Contains some informations about the dataset.

    	训练集、验证集和测试集是通过将用户及其所有交互随机划分为3个集合得到的
    	The training, validation and test sets are obtain by randomly partitioning the users and all their interactions into 3 sets.
    	验证集包含{n_val}用户、test_set {n_test}用户和火车集所有其他用户。
    	The validation set contains {n_val} users, the test_set {n_test} users and the train set all the other users.

    	'''.format(n_val=str(val_set['u'].nunique()), n_test=str(test_set['u'].nunique()))

        results_readme = '''
    	结果文件的格式如下
    	The format of the results file is the following
    	每一行对应一个模型，字段为:
    	Each line correspond to one model, with the fields being:
    		数量的epochs
    		Number of epochs
    		精度
    		precision	
    		sps	
    		user coverage
    		测试集中唯一项的数量
    		number of unique items in the test set
    		建议中唯一项的数量
    		number of unique items in the recommendations
    		成功建议中唯一项的数量
    		number of unique items in the succesful recommendations
    		短期测试集中唯一项的数量(当目标是准确预测下一项时)
    		number of unique items in the short-term test set (when the goal is to predict precisely the next item)
    		数量独特的项目在短期内成功推荐
    		number of unique items in the successful short-term recommendations
    		召回
    		recall
    		NDCG
    	所有指标都计算为“@10”
    	NB: all the metrics are computed "@10"
    	'''
        with open(self.dirname + "data/README", "w") as f:
            f.write(data_readme)
        with open(self.dirname + "results/README", "w") as f:
            f.write(results_readme)