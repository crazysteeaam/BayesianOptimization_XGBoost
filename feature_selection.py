import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import dcor
import concurrent.futures
import time
import seaborn as sns
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.ERROR)

# 全局参数
data_path = 'data1'  # data1含有237个特征（1%方差），data2含有306个（0.5%方差）
select_feature_time = 100  # 筛选特征的次数
top_n = 25  # 选取前n个特征


def dataloader():
    # Load data
    data = pd.read_excel('data/filtered_' + data_path + '.xlsx', index_col=None)
    return data


def get_feature_importance(data):
    # Feature selection
    X = data.iloc[:, 2:]
    Y = data.loc[:, ['pIC50']]
    names = data.columns[2:]

    # 创建随机森林模型求特征重要性
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    feature_importance = pd.DataFrame(
        sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True),
        columns=['importance', 'feature'])
    feature_importance.set_index('feature', inplace=True)

    # 最大互信息数
    mi = mutual_info_regression(X, Y)
    mi = pd.DataFrame(mi, index=X.columns, columns=['mi'])
    mi.index.name = 'feature'

    # 皮尔逊相关系数
    pearson = []
    for i in range(X.shape[1]):
        pearson.append(pearsonr(X.iloc[:, i], Y.iloc[:, 0])[0])
    pearson = pd.DataFrame(pearson, index=X.columns, columns=['pearson'])
    pearson.index.name = 'feature'

    # 距离相关系数
    dcor_list = []
    for i in range(X.shape[1]):
        dcor_list.append(dcor.distance_correlation(np.array(X.iloc[:, i]), np.array(Y.iloc[:, 0])))
    dcor_list = pd.DataFrame(dcor_list, index=X.columns, columns=['dcor'])
    dcor_list.index.name = 'feature'

    # 将四个指标合并到一张表中
    feature_importance = feature_importance.merge(mi, left_index=True, right_index=True)
    feature_importance = feature_importance.merge(pearson, left_index=True, right_index=True)
    feature_importance = feature_importance.merge(dcor_list, left_index=True, right_index=True)

    # 第五个指标是importance*另外三个之和的倒数（perason取绝对值）
    feature_importance['score'] = (
            abs(feature_importance['mi']) +
            abs(feature_importance['pearson']) +
            abs(feature_importance['dcor'])) * feature_importance['importance']

    return feature_importance


def feature_selection(feature_importance):
    # 筛选出score最高的top_n个特征
    selected_feature = feature_importance.sort_values(by='score', ascending=False).iloc[:top_n, :]
    return selected_feature


def feature_selection_worker(i, data):
    feature_importance = get_feature_importance(data)
    selected_feature = feature_selection(feature_importance)
    print("第{}次筛选".format(i + 1))
    print(selected_feature.index)
    return selected_feature.index


def draw_cor_heatmap(data_selected):
    plt.figure(figsize=(30, 20))
    sns.heatmap(data_selected.corr(), annot=True, fmt='.2f')
    # 保存热力图
    plt.savefig('data/export/selected_feature_cor_' + data_path + '_' + str(select_feature_time) + '.png')


def main():
    # Load data
    data = dataloader()
    # Feature selection
    feature_time = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(feature_selection_worker, i, data) for i in range(select_feature_time)]
        for future in concurrent.futures.as_completed(futures):
            selected_features = future.result()
            for feature in selected_features:
                if feature in feature_time:
                    feature_time[feature] += 1
                else:
                    feature_time[feature] = 1
    # 取出频数最高的top_n个特征
    feature_time = pd.DataFrame(feature_time, index=['time']).T
    feature_time.sort_values(by='time', ascending=False, inplace=True)
    selected_feature = feature_time.iloc[:top_n, :]
    # 保存特征
    selected_feature.to_csv('data/export/selected_feature_' + data_path + '_' + str(select_feature_time) + '.csv')
    selected_feature = list(selected_feature.index)

    # # 取出保存的特征
    # selected_feature_time = pd.read_csv(
    #     'data/export/selected_feature_' + data_path + '_' + str(select_feature_time) + '.csv', index_col=0)
    # selected_feature = list(selected_feature_time.index)

    # 根据筛选出的特征取出数据
    data_selected = data.loc[:, selected_feature]

    # 绘制相关性热力图
    draw_cor_heatmap(data_selected)

    # 给data_selected添加pIC50列并调整为第一列
    data_selected.insert(0, 'pIC50', data.loc[:, ['pIC50']])
    data_selected.insert(0, 'SMILES', data.loc[:, ['SMILES']])

    # 得出筛选出top_n个特征的贡献率
    selected_feature_importance = get_feature_importance(data_selected)
    selected_feature_importance.to_csv(
        'data/export/selected_feature_importance_all_' + data_path + '_' + str(select_feature_time) + '.csv')
    # 取出score列，并求占总和的百分比
    selected_feature_importance = selected_feature_importance.loc[:, ['score']]
    selected_feature_importance = round(selected_feature_importance / selected_feature_importance.sum(),5)
    selected_feature_importance.to_csv(
        'data/export/selected_feature_importance_' + data_path + '_' + str(select_feature_time) + '.csv')

    # 保存data_selected
    data_selected.to_csv('data/export/data_selected_' + data_path + '_' + str(select_feature_time) + '.csv')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("用时{}分钟：".format((end - start) / 60))
