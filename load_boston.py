import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report


def mylinear():

    """
    利用线性回归的方法预测波士顿房价
    :return: None
    """

    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 进行标准化处理(特征值和目标值都需要进行转换, 实例化两个标准化API, 再用inverse_transform在转换成之前的值)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测
    # 正规方程求解方式预测结果
    lr = LinearRegression()

    lr.fit(x_train, y_train)
    print(lr.coef_)

    # 预测测试集房子的价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程测试集中每个房子的预测价格: ", y_lr_predict)
    print("正规方程的均方误差为: ", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # 通过梯度下降进行房价预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("SGD coef: ", sgd.coef_)

    # 预测测试集的房子价格
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降测试集中每个房子的预测价格: ", y_sgd_predict)
    print("梯度下降的均方误差为: ", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归预测房子价格
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print("岭回归权重参数为: ", rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print("岭回归测试集中每个房子的价格为: ", y_rd_predict)
    print("岭回归的均方误差为: ", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None

def logistic():
    """
    逻辑回归做二分类进行癌症预测(根据细胞的特征)
    :return: None
    """
    # 数据共有11列，构造每列标签名称
    column_names = ['Sample code number', 'id number', 'Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape Marginal Adhesion', 'Single Epithelial Cell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)

    # 对缺失值进行处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25)

    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0, penalty='l2')
    lg.fit(x_train, y_train)

    print(lg.coef_)
    print("准确率为: ", lg.score(x_test, y_test))

    y_predict = lg.predict(x_test)
    print("召回率: ", classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))
    return None


if __name__ == "__main__":
    mylinear()
