import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from bayes_opt import BayesianOptimization, UtilityFunction
from hyperopt import hp, tpe, fmin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


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


def bho_gbdt_train(X_train, X_test, y_train, y_test):
    # 建立模型
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # 参数空间
    param_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    }

    # 定义优化函数
    def objective(params):
        params = {'n_estimators': int(params['n_estimators']),
                  'max_depth': int(params['max_depth']),
                  'learning_rate': params['learning_rate'],
                  'subsample': params['subsample'],
                  'min_samples_split': int(params['min_samples_split']),
                  'min_samples_leaf': int(params['min_samples_leaf'])}
        model.set_params(**params)
        score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1,
                                scoring=make_scorer(mean_squared_error, greater_is_better=False)).mean()
        return -score

    # 进行贝叶斯优化
    best = fmin(fn=objective, space=param_space, max_evals=100, rstate=np.random.default_rng(42), algo=tpe.suggest)
    best_params = {'n_estimators': int(best['n_estimators']),
                   'max_depth': int(best['max_depth']),
                   'learning_rate': best['learning_rate'],
                   'subsample': best['subsample'],
                   'min_samples_split': int(best['min_samples_split']),
                   'min_samples_leaf': int(best['min_samples_leaf'])}

    # 使用最优参数训练模型
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # 测试模型
    print(model.score(X_test, y_test))
    # 评估预测结果
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print("优化后的GBDT 训练集RMSE: %f" % train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print("优化后的GBDT 测试集RMSE: %f" % test_rmse)
    # R2
    print("优化后的GBDT 训练集R2: %f" % r2_score(y_train, model.predict(X_train)))
    print("优化后的GBDT 测试集R2: %f" % r2_score(y_test, model.predict(X_test)))
    # MAE
    print("优化后的GBDT 训练集MAE: %f" % mean_absolute_error(y_train, model.predict(X_train)))
    print("优化后的GBDT 测试集MAE: %f" % mean_absolute_error(y_test, model.predict(X_test)))


def stoa_gbdt_train(X_train, X_test, y_train, y_test):
    # 参数空间
    param_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', 0, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    }

    bounds = [(100, 1000), (1, 10), (0, 1), (0.5, 1), (2, 10), (1, 10)]

    # 定义优化函数
    def objective(params):
        params = {'n_estimators': int(params['n_estimators']),
                  'max_depth': int(params['max_depth']),
                  'learning_rate': params['learning_rate'],
                  'subsample': params['subsample'],
                  'min_samples_split': int(params['min_samples_split']),
                  'min_samples_leaf': int(params['min_samples_leaf'])}
        model.set_params(**params)
        score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1,
                                scoring=make_scorer(mean_squared_error, greater_is_better=False)).mean()
        return -score

    # 初始化参数
    n_etimators, max_depth, learning_rate, subsample, min_samples_split, min_samples_leaf = 100, 3, 0.1, 0.5, 2, 1

    # 初始化种群
    pop_size = 20
    population = []
    for _ in range(pop_size):
        individual = []
        for bound in bounds:
            individual.append(np.random.uniform(low=bound[0], high=bound[1]))
        population.append(individual)

    model = GradientBoostingRegressor(n_estimators=int(individual[0]), max_depth=int(individual[1]),
                                      learning_rate=individual[2],
                                      subsample=individual[3], min_samples_split=int(individual[4]),
                                      min_samples_leaf=int(individual[5]), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # RMSE
    print("GBDT 训练集RMSE: %f" % np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    print("GBDT 测试集RMSE: %f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    # R2
    print("GBDT 训练集R2: %f" % r2_score(y_train, model.predict(X_train)))
    print("GBDT 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("GBDT 训练集MAE: %f" % mean_absolute_error(y_train, model.predict(X_train)))
    print("GBDT 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))

    # 迭代优化
    max_gen = 100
    for gen in tqdm(range(max_gen)):
        # 计算适应度
        fitness = [objective({'n_estimators': int(individual[0]), 'max_depth': int(individual[1]),
                              'learning_rate': individual[2], 'subsample': individual[3],
                              'min_samples_split': int(individual[4]), 'min_samples_leaf': int(individual[5])}) for
                   individual
                   in population]
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
                fitness_v = objective({'n_estimators': int(v[0]), 'max_depth': int(v[1]),
                                       'learning_rate': v[2], 'subsample': v[3],
                                       'min_samples_split': int(v[4]), 'min_samples_leaf': int(v[5])})
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
            fitness_v = objective({'n_estimators': int(v[0]), 'max_depth': int(v[1]),
                                   'learning_rate': v[2], 'subsample': v[3],
                                   'min_samples_split': int(v[4]), 'min_samples_leaf': int(v[5])})
            # 更新个体
            if fitness_v < fitness[i]:
                population[i] = v
                fitness[i] = fitness_v
        # if (gen + 1) % 10 == 0:
        print("Iteration: {}, Best Fitness: {:.6f}".format(gen, best_fitness))
        print("Current Parameters: {}".format(best_solution))

    # 输出最优解
    print("Best Solution: {}".format(best_solution))
    print("Best Fitness: {}".format(best_fitness))

    # 训练模型
    model = GradientBoostingRegressor(n_estimators=int(best_solution[0]), max_depth=int(best_solution[1]),
                                      learning_rate=best_solution[2],
                                      subsample=best_solution[3], min_samples_split=int(best_solution[4]),
                                      min_samples_leaf=int(best_solution[5]), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 评估模型
    # RMSE
    print("GBDT 训练集RMSE: %f" % np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    print("GBDT 测试集RMSE: %f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    # R2
    print("GBDT 训练集R2: %f" % r2_score(y_train, model.predict(X_train)))
    print("GBDT 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("GBDT 训练集MAE: %f" % mean_absolute_error(y_train, model.predict(X_train)))
    print("GBDT 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


def stoa_xgb_train(X_train, X_test, y_train, y_test):
    bounds = [(3, 20), (0, 1), (0.5, 1)]

    global dtrain
    # 转化为xgb的DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)

    xgb_evaluate(5, 0.5, 0.5)

    # 初始化种群
    pop_size = 20
    population = []
    for _ in range(pop_size):
        individual = []
        for bound in bounds:
            individual.append(np.random.uniform(low=bound[0], high=bound[1]))
        population.append(individual)

    # 初始化参数
    max_depth, gamma, learning_rate = int(individual[0]), individual[1], individual[2]
    xgbmodel = XGBRegressor(max_depth=max_depth, gamma=gamma, learning_rate=learning_rate, objective='reg:squarederror')
    xgbmodel.fit(X_train, y_train)
    y_pred = xgbmodel.predict(X_test)
    print("XGBoost 训练集R2: %f" % r2_score(y_train, xgbmodel.predict(X_train)))
    print("XGBoost 测试集R2: %f" % r2_score(y_test, y_pred))

    # 迭代优化
    max_gen = 30
    for gen in tqdm(range(max_gen)):
        # 计算适应度
        fitness = [xgb_evaluate(int(individual[0]), individual[1], individual[2]) for individual in population]

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
                fitness_v = xgb_evaluate(int(v[0]), v[1], v[2])
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
            fitness_v = xgb_evaluate(int(v[0]), v[1], v[2])
            # 更新个体
            if fitness_v < fitness[i]:
                population[i] = v
                fitness[i] = fitness_v
        # if (gen + 1) % 10 == 0:
        print("Iteration: {}, Best Fitness: {:.6f}".format(gen, best_fitness))
        print("Current Parameters: {}".format(best_solution))

    # 输出最优解
    print("Best Solution: {}".format(best_solution))
    print("Best Fitness: {}".format(best_fitness))

    # 训练模型
    xgbmodel = XGBRegressor(max_depth=int(best_solution[0]), gamma=best_solution[1], learning_rate=best_solution[2],
                            objective='reg:squarederror')
    xgbmodel.fit(X_train, y_train)
    y_pred = xgbmodel.predict(X_test)
    # 评估模型
    # RMSE
    print("XGBoost 训练集RMSE: %f" % mean_squared_error(y_train, xgbmodel.predict(X_train), squared=False))
    print("XGBoost 测试集RMSE: %f" % mean_squared_error(y_test, y_pred, squared=False))
    # R2
    print("XGBoost 训练集R2: %f" % r2_score(y_train, xgbmodel.predict(X_train)))
    print("XGBoost 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("XGBoost 训练集MAE: %f" % mean_absolute_error(y_train, xgbmodel.predict(X_train)))
    print("XGBoost 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


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


def ga_bho_optimization(X_train, X_test, y_train, y_test):
    # 定义超参数空间的边界和初始采样点数量
    bounds = [(0, 5), (0, 10), (0, 1)]
    initial_samples = 5

    # 定义目标函数
    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    # 贝叶斯模型初始化
    X = np.array([])
    y = np.array([])

    # 初始采样（步骤1）
    for _ in range(initial_samples):
        sample = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        X = np.append(X, [sample])
        print(X.shape)
        y = np.append(y, objective_function(sample))
        print(y.shape)

    X = X.reshape(-1, len(bounds))

    print(X.shape, y.shape)

    # 高斯过程回归模型初始化
    kernel = Matern(length_scale=1.0, nu=2.5)
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

    # 最大迭代次数
    max_iterations = 10

    # 进行迭代
    for iteration in range(max_iterations):
        # 拟合高斯过程模型（步骤2）
        model.fit(X, y)

        # 使用贝叶斯优化算法的代理模型预测适应度（步骤2）
        mu, sigma = model.predict(X, return_std=True)

        # 选择操作（步骤3）
        selected_indices = np.where(mu >= np.percentile(mu, 50))[0]
        selected_X = X[selected_indices]

        # 遗传操作（步骤4）
        children = []
        while len(children) < initial_samples:
            parent1, parent2 = random.choices(selected_X, k=2)
            child = [random.choice(dim) for dim in zip(parent1, parent2)]
            children.append(child)

        # 评估新生成的子代并计算目标值（步骤5）
        for child in children:
            X = np.vstack((X, child))
            y = np.append(y, objective_function(child))

        # 更新贝叶斯模型（步骤6）
        model.fit(X, y)

    # 输出最优超参数和目标值
    best_index = np.argmax(y)
    best_hyperparameters = X[best_index]
    best_objective_value = y[best_index]
    print("Best hyperparameters:", best_hyperparameters)
    print("Best objective value:", best_objective_value)

    # 使用最优超参数训练模型
    model = XGBRegressor(n_estimators=int(best_hyperparameters[0]),
                         max_depth=int(best_hyperparameters[1]),
                         learning_rate=best_hyperparameters[2])
    model.fit(X_train, y_train)
    # 使用训练好的模型进行预测
    x_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    # 评估预测结果
    train_rmse = np.sqrt(mean_squared_error(y_train, x_pred))
    print("GA_BHO 训练集RMSE: %f" % train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("GA_BHO 测试集RMSE: %f" % test_rmse)
    # R2
    print("GA_BHO 训练集R2: %f" % r2_score(y_train, x_pred))
    print("GA_BHO 测试集R2: %f" % r2_score(y_test, y_pred))
    # MAE
    print("GA_BHO 训练集MAE: %f" % mean_absolute_error(y_train, x_pred))
    print("GA_BHO 测试集MAE: %f" % mean_absolute_error(y_test, y_pred))


def main():
    data = data_loader()
    X = data.iloc[:, 2:]
    Y = data.loc[:, ['pIC50']]
    # X = X.drop(['minHsOH', 'VC-5', 'SsOH', 'MLogP', 'maxsOH'], axis=1)
    X = X.drop(['SsOH', 'maxHsOH', 'minHsOH', 'minsOH', 'maxsOH'], axis=1)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
    # normal_xgb_train(X_train, X_test, y_train, y_test)
    # bayesian_optimized_xgb_train(X_train, X_test, y_train, y_test)
    # svr_train(X_train, X_test, y_train, y_test)
    # gbdt_train(X_train, X_test, y_train, y_test)
    # cnn_lstm_train(X_train, X_test, y_train, y_test)
    # ga_bho_optimization(X_train, X_test, y_train, y_test)
    # bho_gbdt_train(X_train, X_test, y_train, y_test)
    # stoa_xgb_train(X_train, X_test, y_train, y_test)
    stoa_gbdt_train(X_train, X_test, y_train, y_test)
    # predict()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("用时{}分钟：".format((end_time - start_time) / 60))
