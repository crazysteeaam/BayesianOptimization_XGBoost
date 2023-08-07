import random

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 定义超参数空间的边界和初始采样点数量
bounds = [(0, 5), (0, 10), (0, 1)]
initial_samples = 5


# 定义目标函数
def objective_function(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


# 这里只是假设你的X数据有3个特征，你应该用你自己的数据替换这个
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5]
])

# 这里只是假设你的y数据是一维的，你应该用你自己的数据替换这个
y = np.array([2.3, 2.6, 3.1, 3.6, 4.1])

# 初始采样（步骤1）
for _ in range(initial_samples):
    sample = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    X = np.append(X, [sample])
    y = np.append(y, objective_function(sample))

X = X.reshape(-1, len(bounds))

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
