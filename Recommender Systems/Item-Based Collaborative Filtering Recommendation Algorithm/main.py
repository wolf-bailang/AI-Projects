
from item_baesd_cf_model import ItemBasedCF
from data_pre_process import DataPreProcess

if __name__ == '__main__':
    # 文件路径
    rating_file = './data/ratings.csv'

    data_set = DataPreProcess().get_dataset(rating_file)        # 先实例化类，再读文件得到“用户-电影”数据
    itemCF = ItemBasedCF(data_set)      # 实例化
    itemCF.calc_movie_sim()     # 计算电影之间的相似度
    itemCF.evaluate()       # 产生推荐并通过准确率、召回率和覆盖率进行评估
