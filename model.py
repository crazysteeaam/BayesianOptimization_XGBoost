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
    data = pd.read_excel('data/test.xlsx',index_col=0)
    # 调整列顺序MDEC-23,MDEO-12,maxssO,minHBa,VCH-5,maxdssC,VC-5,mindssC,minHsOH,ETA_Shape_Y,maxHsOH,minHBint5,minsssN,maxHBint8,LipoaffinityIndex,BCUTc-1h,ATSc3,ETA_BetaP_s,ETA_EtaP_F,mindsCH
    data = data.loc[:,['MDEC-23', 'MDEO-12', 'maxssO', 'minHBa', 'VCH-5', 'maxdssC', 'VC-5', 'mindssC', 'minHsOH', 'ETA_Shape_Y', 'maxHsOH', 'minHBint5', 'minsssN', 'maxHBint8', 'LipoaffinityIndex', 'BCUTc-1h', 'ATSc3', 'ETA_BetaP_s', 'ETA_EtaP_F', 'mindsCH']]
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
    # 定义搜索空间
    bounds = [(-5, 5), (-5, 5), (-5, 5), (0, 10)]

    # 定义适应度函数
    def fitness_function(params):
        C, gamma, coef0, degree = params
        clf = SVR(kernel='rbf', C=10 ** C, gamma=10 ** gamma, coef0=10 ** coef0, degree=int(degree))
        clf.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))
        return -rmse

    fitness_function([0, 0, 0, 0])

    # 初始化种群
    pop_size = 20
    population = []
    for i in range(pop_size):
        individual = []
        for bound in bounds:
            individual.append(np.random.uniform(low=bound[0], high=bound[1]))
        population.append(individual)

    # 初始mse和r2
    C, gamma, coef0, degree = individual[0], individual[1], individual[2], individual[3]
    clf = SVR(kernel='rbf', C=10 ** C, gamma=10 ** gamma, coef0=10 ** coef0, degree=int(degree))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(y_pred)

    # 评估预测结果
    # rmse
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("初始RMSE: %f" % (rmse))

    # 迭代优化
    max_gen = 200
    for gen in tqdm(range(max_gen)):
        # 计算适应度
        fitness = [fitness_function(individual) for individual in population]

        # 找到最优解
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # 移动和聚集
        for i in range(pop_size):
            if i != best_idx:
                # print(best_solution)
                # print(population[i])
                dx = []
                for j in range(len(bounds)):
                    dx.append(best_solution[j] - population[i][j])
                # dx = best_solution - population[i]
                # print(np.dot(np.random.uniform(), dx))
                # 添加柯西变异
                cauchy = np.random.standard_cauchy(len(bounds))
                # 添加高斯变异
                gaussian = np.random.normal(0, 1, len(bounds))
                v = (population[i] + np.dot(np.random.uniform(), dx)) * (
                        1 + (1 - gen ** 2 / max_gen ** 2) * cauchy + gen ** 2 / max_gen ** 2 * gaussian)
                # 边界处理
                # print(v)
                # print(bounds[0])
                v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
                # 计算适应度
                fitness_v = fitness_function(v)
                # 更新个体
                if fitness_v < fitness[i]:
                    population[i] = v
                    fitness[i] = fitness_v

        # 生成新的个体
        for i in range(pop_size):
            idx1, idx2, idx3 = np.random.choice(pop_size, size=3, replace=False)
            r1, r2, r3 = population[idx1], population[idx2], population[idx3]
            r1 = np.array(r1)
            r2 = np.array(r2)
            r3 = np.array(r3)
            # 添加柯西变异
            cauchy = np.random.standard_cauchy(len(bounds))
            # 添加高斯变异
            gaussian = np.random.normal(0, 1, len(bounds))
            v = (r1 + 0.5 * (r2 - r3)) * (
                    1 + (1 - gen ** 2 / max_gen ** 2) * cauchy + gen ** 2 / max_gen ** 2 * gaussian)
            # 边界处理
            v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
            # 计算适应度
            fitness_v = fitness_function(v)
            # 更新个体
            if fitness_v < fitness[i]:
                population[i] = v
                fitness[i] = fitness_v
        # if (gen + 1) % 10 == 0:
        print("Iteration: {}, Best Fitness: {:.6f}".format(gen, best_fitness))

    # 输出最优解
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)

    # 预测结果
    C, gamma, coef0, degree = best_solution
    clf = SVR(kernel='rbf', C=10 ** C, gamma=10 ** gamma, coef0=10 ** coef0, degree=int(degree))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)

    # 评估预测结果
    # rmse
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("优化后的RMSE: %f" % (rmse))


def main():
    # data = data_loader()
    # X = data.iloc[:, 2:]
    # Y = data.loc[:, ['pIC50']]
    #
    # # X筛除掉BCUTc-1l,SPC=6,ShsOH,minsOH,maxHBint5这五列
    # X = X.drop(['BCUTc-1l', 'SPC-6', 'SHsOH', 'minsOH', 'maxHBint5'], axis=1)
    #
    # # 划分数据集
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
    # # normal_xgb_train(X_train, X_test, y_train, y_test)
    # bayesian_optimized_xgb_train(X_train, X_test, y_train, y_test)
    # # svr_train(X_train, X_test, y_train, y_test)
    predict()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("用时{}分钟：".format((end_time - start_time) / 60))
