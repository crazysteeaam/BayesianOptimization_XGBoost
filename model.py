import random
import time

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential
from tqdm import tqdm


def data_loader():
    data = pd.read_csv('data/export/data_selected_data1_100.csv', index_col=0)
    return data


def xgb_evaluate(max_depth, gamma, colsample_bytree):
    # 定义优化目标
    params = {'eval_metric': 'rmse',  # 评估指标是均方根误差
              'max_depth': int(max_depth),  # 每棵树的最大深度
              'subsample': 0.8,  # 每棵树对样本的采样率
              'eta': 0.2,  # 学习率
              'gamma': gamma,  # 每棵树的叶子节点上所有样本的最小损失函数下降值
              'colsample_bytree': colsample_bytree  # 每棵树的特征采样率
              }
    # 使用xgb.cv进行交叉验证
    cv_result = xgb.cv(params,
                       dtrain,
                       num_boost_round=100,  # 迭代次数
                       nfold=3)  # 交叉验证的折数

    # 我们想要最小化均方根误差，所以使用负值
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


def bayesian_optimized_xgb_train(X_train, X_test, y_train, y_test):
    global dtrain
    # 转化为xgb的DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 定义优化参数
    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                                 'gamma': (0, 1),
                                                 'colsample_bytree': (0.5, 1)})

    # 先进行5次随机初始化
    init_points = 5
    for _ in range(init_points):
        params = {'max_depth': random.randint(3, 10),
                  'gamma': random.uniform(0, 1),
                  'colsample_bytree': random.uniform(0.5, 1)}
        target = xgb_evaluate(**params)
        xgb_bo.register(params=params, target=target)

    # 使用贝叶斯优化进行参数优化
    util_func = UtilityFunction(kind="poi",  # pi: probability of improvement
                                kappa=10,  # kappa越大，越注重探索
                                xi=5  # xi越大，越注重利用
                                )

    for i in tqdm(range(100)):
        # 使用贝叶斯优化进行参数优化
        next_point_to_probe = xgb_bo.suggest(util_func)
        # 在每个参数上添加一个小的高斯扰动
        next_point_to_probe = {k: v + np.random.normal(scale=0.001) for k, v in next_point_to_probe.items()}
        target = xgb_evaluate(**next_point_to_probe)
        xgb_bo.register(params=next_point_to_probe, target=target)
        print(xgb_bo.max['params'])
        print(f"iteration {i + 1}, target {target}")

    # 输出最优参数
    params = xgb_bo.max['params']
    print(params)

    # 使用最优参数进行训练
    params['max_depth'] = int(params['max_depth'])
    model = xgb.train(params, dtrain, num_boost_round=100)

    # 测试集预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(dtrain)))
    print("优化后的训练集RMSE: %f" % (train_rmse))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("优化后的测试集RMSE: %f" % (test_rmse))
    # R2
    print("优化后的训练集R2: %f" % r2_score(y_train, model.predict(dtrain)))
    print("优化后的测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("优化后的训练集MAE: %f" % mean_absolute_error(y_train, model.predict(dtrain)))
    print("优化后的测试集MAE: %f" % mean_absolute_error(y_test, y_pred))
    # 保存模型文件
    model.save_model('model/xgb.model')


def predict():
    # 加载模型文件
    model = xgb.Booster(model_file='model/xgb.model')
    # 预测
    data = pd.read_excel('data/test.xlsx', index_col=0)
    # 调整列顺序MDEC-23,MDEO-12,maxssO,minHBa,VCH-5,maxdssC,VC-5,mindssC,minHsOH,ETA_Shape_Y,maxHsOH,minHBint5,minsssN,maxHBint8,LipoaffinityIndex,BCUTc-1h,ATSc3,ETA_BetaP_s,ETA_EtaP_F,mindsCH
    data = data.loc[:,
           ['MDEC-23', 'MDEO-12', 'maxssO', 'minHBa', 'VCH-5', 'maxdssC', 'VC-5', 'mindssC', 'minHsOH', 'ETA_Shape_Y',
            'maxHsOH', 'minHBint5', 'minsssN', 'maxHBint8', 'LipoaffinityIndex', 'BCUTc-1h', 'ATSc3', 'ETA_BetaP_s',
            'ETA_EtaP_F', 'mindsCH']]
    print(data)
    dtest = xgb.DMatrix(data)
    y_pred = model.predict(dtest)
    data['y_pred'] = y_pred
    data.to_excel('data/result.xlsx')


def normal_xgb_train(X_train, X_test, y_train, y_test):
    # 初始化XGBoost回归模型
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',  # 使用均方误差作为损失函数
                              colsample_bytree=0.3,  # 每棵树的特征采样率
                              learning_rate=0.1,  # 学习率
                              max_depth=5,  # 每棵树的最大深度
                              alpha=10,  # L1正则化系数
                              n_estimators=10  # 树的数量
                              )
    # 拟合训练数据
    xg_reg.fit(X_train, y_train)
    # 预测测试集
    y_pred = xg_reg.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, xg_reg.predict(X_train)))
    print("未优化训练集RMSE: %f" % train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("未优化测试集RMSE: %f" % test_rmse)
    # R2
    print("未优化训练集R2: %f" % r2_score(y_train, xg_reg.predict(X_train)))
    print("未优化测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("未优化训练集MAE: %f" % mean_absolute_error(y_train, xg_reg.predict(X_train)))
    print("未优化测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


def svr_train(X_train, X_test, y_train, y_test):
    # 实例化一个SVR模型
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # 用训练数据训练模型
    svr.fit(X_train, y_train)
    # 预测测试数据
    y_pred = svr.predict(X_test)
    # 评估预测结果
    train_rmse = np.sqrt(mean_squared_error(y_train, svr.predict(X_train)))
    print("SVR 训练集RMSE: %f" % train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("SVR 测试集RMSE: %f" % test_rmse)
    # R2
    print("SVR 训练集R2: %f" % r2_score(y_train, svr.predict(X_train)))
    print("SVR 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("SVR 训练集MAE: %f" % mean_absolute_error(y_train, svr.predict(X_train)))
    print("SVR 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


def gbdt_train(X_train, X_test, y_train, y_test):
    # GBDT
    # 建立模型
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估预测结果
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print("GBDT 训练集RMSE: %f" % train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("GBDT 测试集RMSE: %f" % test_rmse)
    # R2
    print("GBDT 训练集R2: %f" % r2_score(y_train, model.predict(X_train)))
    print("GBDT 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("GBDT 训练集MAE: %f" % mean_absolute_error(y_train, model.predict(X_train)))
    print("GBDT 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.pool1 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x, _ = self.lstm(x.transpose(1, 2))
        x = self.fc(x[:, -1, :])
        return x


def cnn_lstm_train(X_train, X_test, y_train, y_test):
    # CNN-LSTM
    # 重构数据
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_train = y_train.values.reshape((y_train.shape[0], 1))
    y_test = y_test.values.reshape((y_test.shape[0], 1))
    # 转换成torch的tensor
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    # 打印数据形状
    print(X_train.shape, y_train.shape)
    # 建立模型
    model = CNN_LSTM()
    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    for epoch in tqdm(range(3000)):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    # 使用训练好的模型进行预测
    with torch.no_grad():
        x_pred = model(X_train)
        y_pred = model(X_test)
        # 评估预测结果
        train_rmse = np.sqrt(mean_squared_error(y_train, x_pred))
        print("CNN_LSTM 训练集RMSE: %f" % train_rmse)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("CNN_LSTM 测试集RMSE: %f" % test_rmse)
        # R2
        print("CNN_LSTM 训练集R2: %f" % r2_score(y_train, x_pred))
        print("CNN_LSTM 测试集R2: %f" % r2_score(y_test, y_pred))
        # MAE
        print("CNN_LSTM 训练集MAE: %f" % mean_absolute_error(y_train, x_pred))
        print("CNN_LSTM 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


def main():
    data = data_loader()
    X = data.iloc[:, 2:]
    Y = data.loc[:, ['pIC50']]
    X = X.drop(['minHsOH', 'VC-5', 'SsOH', 'MLogP', 'maxsOH'], axis=1)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
    # normal_xgb_train(X_train, X_test, y_train, y_test)
    # bayesian_optimized_xgb_train(X_train, X_test, y_train, y_test)
    # svr_train(X_train, X_test, y_train, y_test)
    # gbdt_train(X_train, X_test, y_train, y_test)
    cnn_lstm_train(X_train, X_test, y_train, y_test)
    # predict()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("用时{}分钟：".format((end_time - start_time) / 60))
