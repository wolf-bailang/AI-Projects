from __future__ import print_function

import numpy as np
from data_preprocess import DataPreprocess

def main():
    # 数据处理
    # 文件夹路径
    # dirname = 'D:/zjb/data/'
    # 文件名
    filename = 'D:\\zjb\\data1\\ratings.csv'
    # 实例化类
    datapreprocess = DataPreprocess(filename)
    # args = command_parser()
    np.random.seed(seed=1)
    # 询问用户是否确定要在该目录中创建文件。
    datapreprocess.warn_user()
    datapreprocess.create_dirs()
    # 从文件名加载数据，并根据时间戳对其排序。
    data = datapreprocess.load_data()
    print(data.shape)
    # 删除出现在太少交互中的用户和项。
    data = datapreprocess.remove_rare_elements(data)
    # 在dirname中将原始用户id和项id映射为连续的数字id。
    data = datapreprocess.save_index_mapping(data)
    # 将数据集分为训练集、验证集和测试集。
    train_set, val_set, test_set = datapreprocess.split_data(data)
    # 以序列格式转换训练/验证/测试集并保存它们。
    datapreprocess.make_sequence_format(train_set, val_set, test_set)
    datapreprocess.save_data_stats(data, train_set, val_set, test_set)
    datapreprocess.make_readme(val_set, test_set)
    print('Data ready!')
    rint(data.head(10))

    # 构建模型

    # 训练模型

    # 预测结果


if __name__ == '__main__':
    main()


