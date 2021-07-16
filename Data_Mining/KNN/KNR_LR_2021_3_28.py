# encoding=utf-8

# Time : 2021/3/28 13:18 

# Author : 啵啵

# File : KNR_LR_2021_3_28.py

# Software: PyCharm


class KNR_LR(object):
    """Iris of KNeighborsClassifier and LinearRegression

    args:
        None

    returns:
        X_train: 训练集的X维度
        X_test: 测试集的X维度
        y_train: 训练集的y维度
        y_test: 测试集的y维度
        -------------------------
        第三题(question three)
        y_predict: KNR(KNeighborsRegressor models)模型的预测值
        X_KNR_test: KNR(KNeighborsRegressor models)模型的用于预测的X数据集
        y_KNR_test: KNR(KNeighborsRegressor models)模型的用于预测的Y数据集
        -------------------------
        -------------------------
        计算KNR,LR模型准确率
        y_KNR_predict: KNR(KNeighborsRegressor models)模型的预测值
        y_LR_predict: LR(LinearRegression models)模型的预测值
        -------------------------
        original_data: iris原始数据集
        original_target: iris特征量化
        iris_data: 经过ETL之后新的用于模型训练测试的数据集
        KNR: KNR(KNeighborsRegressor models)模型
        LR: LR(LinearRegression models)模型

    functions:
        __load_iris_data: 加载读取数据(load iris data)
        __transform_iris_data: 预处理数据(ETL iris data)
        __split_iris_data： 数据集切分(split iris data)
        __fit_KNR_model: 训练KNR模型(fit KNR models)
        __fit_LR_model: 训练LR模型(fit LR models)
        __predict_KNR_LR_model: 预测KNR和LR模型值(predict the value of KNR and LR)
        __accuracy_KNR_LR_model: 计算准确度(calculate the accuracy of the KNR models and LR models)
        __predict_petal_width: 预测花瓣宽度(predict the width of the petal)
        __calculate_error: 计算曼哈顿距离,欧氏距离,闵可夫斯基距离(calculate the PManhattan,Euclidian and Minkowski distance of the KNN models)

    Raises:
        None

    """

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_predict = None
        self.X_KNR_test = None
        self.y_KNR_test = None

        self.y_KNR_predict = None
        self.y_LR_predict = None

        self.original_data = None
        self.original_target = None
        self.iris_data = None

        self.KNR = None
        self.LR = None

    # Load iris data
    def __load_iris_data(self):
        from sklearn.datasets import load_iris

        __iris = load_iris()
        self.original_data = __iris.data
        self.original_target = __iris.target

    # ETL data
    def __transform_iris_data(self):
        import pandas as pd
        original_data_df = pd.DataFrame(self.original_data,
                                        columns=["sepal length(cm)",
                                                 "sepal width(cm)",
                                                 "petal length(cm)",
                                                 "petal width(cm)"])
        original_target_df = pd.DataFrame(columns=["species"], data=self.original_target)

        # 按列合并数据集
        self.iris_data = pd.concat([original_data_df, original_target_df], axis=1)

    # first question: split the iris dataset into 70% train data and 30% test data
    def __split_iris_data(self):
        from sklearn.model_selection import train_test_split
        # 生成X,y用于切分数据集
        X = self.iris_data.drop("petal width(cm)", axis=1)
        y = self.iris_data["petal width(cm)"]
        self.X_train, self.X_test, \
        self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, shuffle=True) # random_sate:确保每次随机取到的数值一样(可选)

    # Fit LinearRegression models
    def __fit_LR_model(self):
        from sklearn.linear_model import LinearRegression
        self.LR = LinearRegression()
        self.LR.fit(self.X_train, self.y_train)

    # Fit KNeighborsRegressor models
    def __fit_KNR_model(self):
        from sklearn.neighbors import KNeighborsRegressor
        self.KNR = KNeighborsRegressor(n_neighbors=8)
        self.KNR.fit(self.X_train, self.y_train)

    # Predict models
    def __predict_KNR_LR_model(self):
        self.y_KNR_predict = self.KNR.predict(self.X_test)
        self.y_LR_predict = self.LR.predict(self.X_test)

    # Second question: calculate Accuracy
    def __accuracy_KNR_LR_model(self):
        from sklearn.metrics import explained_variance_score
        self.KNR_Accuracy = explained_variance_score(self.y_test, self.y_KNR_predict)
        self.LR_Accuracy = explained_variance_score(self.y_test, self.y_LR_predict)
        print("KNR_Accuracy:", self.KNR_Accuracy, "\n")
        print("LR_Accuracy:", self.LR_Accuracy, "\n")
        return self

    def __predict_petal_width(self):
        import pandas as pd

        data = {
            "sepal length(cm)": [5.4],
            "sepal width(cm)": [3.7],
            "petal length(cm)": [1.5],
            "petal width(cm)": [0.3],
            "species": 0,
        }
        test_data = pd.DataFrame(data)
        self.X_KNR_test = test_data.drop("petal width(cm)", axis=1)
        self.y_KNR_test = test_data["petal width(cm)"]
        self.y_predict = self.KNR.predict(self.X_KNR_test)

    def __calculate_error(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        print("Actual petal width(cm):", 0.3, "\n")
        print("Predicted petal width(cm):", self.y_predict[0], "\n")
        print("Mean Absolute Error:", mean_absolute_error(self.y_KNR_test, self.y_predict), "\n")
        print("Mean Squared Error:", mean_squared_error(self.y_KNR_test, self.y_predict), "\n")
        print("Mean Root Squared Error:", np.sqrt(mean_squared_error(self.y_KNR_test, self.y_predict)), "\n")

    # 允许类的实例像函数一样被调用
    def __call__(self, *args, **kwargs):
        # Load data
        self.__load_iris_data()

        # ETL data
        self.__transform_iris_data()

        # Split data
        self.__split_iris_data()

        # Fit LR
        self.__fit_LR_model()

        # Fit KNR
        self.__fit_KNR_model()

        # predict KNR LR data
        self.__predict_KNR_LR_model()

        # Accuracy
        self.__accuracy_KNR_LR_model()

        # Predict petal width
        self.__predict_petal_width()

        # Manhattan Euclidian Minkowski distance
        self.__calculate_error()

if __name__ == "__main__":
    KNR_LR()()
