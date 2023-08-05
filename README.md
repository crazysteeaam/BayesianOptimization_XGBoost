
## 安装依赖环境
``` bash
pip install -r requirements.txt
```

## 运行特征筛选
运行前先调整 `feature_selection.py` 中的参数

``` python
data_path = 'data1'  # data1含有237个特征（1%方差），data2含有306个（0.5%方差）
select_feature_time = 100  # 筛选特征的次数
top_n = 25  # 选取前n个特征
```

运行 `feature_selection.py` 文件
``` bash
python feature_selection.py
```
得到特征筛选后的数据 `data/export/data_selected_{data_path}_{select_feature_time}.csv` 文件

筛选的20个特征的贡献率数据 `data/export/selected_feature_importance_{data_path}_{select_feature_time}.csv` 文件

运行时间3.5分钟左右（测试设备：MacBook Pro 14 2021 M1 Pro）

## 运行模型

运行前先调整 `model.py` 中的参数

``` python

```

```bash
python model.py
```