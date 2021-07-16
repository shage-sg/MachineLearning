# encoding=utf-8

# Time : 2021/3/28 10:15 

# Author : 啵啵

# File : KNC_2021_3_28.py

# Software: PyCharm


import logging


class KNC(object):
    """Iris of KNeighborsClassifier

    args:
        None

    returns:
        X_train: 训练集的X维度
        X_test: 测试集的X维度
        y_train: 训练集的y维度
        y_test: 测试集的y维度
        y_predict: 模型的预测值
        iris_data: iris数据集
        iris_target: iris特征量化
        iris_table: iris表格化展示(show iris data by using table)
        KNC: KNC模型(KNeighborsClassifier)

    functions:
        __load_iris_data: 加载读取数据(load iris data)
        __show_iris_data: 以表格的形式展示数据(show iris data)
        __describe_iris_data: iris数据的描述信息(describe iris data)
        __split_iris_data： 数据集切分(split iris data)
        __fit_KNC_model: 训练KNC模型(fit KNC models)
        __predict_KNC_model: 预测KNC模型值(predict KNC value)
        __accuracy_KNC_model: 计算准确度(calculate the accuracy of the KNC models)
        __confusion_matrix_model: 计算混淆矩阵(calculate the Confusion Matrix of the KNC models)
        __report_model: 计算Precision, recall and F1 score值(calculate the Precision, recall and F1 score of the KNC models)

    Raises:
        None

    """

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_predict = None

        self.iris_data = None
        self.iris_target = None
        self.iris_table = None

        self.KNC = None

    # Load iris data
    def __load_iris_data(self):
        from sklearn.datasets import load_iris
        __iris = load_iris()
        self.iris_data = __iris.data
        self.iris_target = __iris.target

    # second question: The iris data and label as table
    def __show_iris_data(self) -> None:
        import pandas as pd
        self.iris_table = pd.DataFrame(self.iris_data,
                                       columns=["sepal length(cm)",
                                                "sepal width(cm)",
                                                "petal length(cm)",
                                                "petal width(cm)"])

        print("Iris table: \n", self.iris_table, "\n\n\n")

    # four question: Describe information of the table
    def __describe_iris_data(self) -> str:
        iris_describe = self.iris_table.describe()
        print("Iris describe: \n", iris_describe, "\n\n\n")
        return iris_describe

    # first question: split the iris dataset into 80% train data and 20% test data
    def __split_iris_data(self):
        from sklearn.model_selection import train_test_split
        X = self.iris_data
        y = self.iris_target
        self.X_train, self.X_test, \
        self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, shuffle=True)

    # Fit models
    def __fit_KNC_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.KNC = KNeighborsClassifier(n_neighbors=5)
        self.KNC.fit(self.X_train, self.y_train)

    # Predict models
    def __predict_KNC_model(self):
        self.y_predict = self.KNC.predict(self.X_test)

    # Accuracy
    def __accuracy_KNC_model(self):
        from sklearn.metrics import accuracy_score
        Accuracy = accuracy_score(self.y_test, self.y_predict)

        print("Accuracy:", Accuracy, "\n\n\n")
        return Accuracy

    # Confusion matrix
    def __confusion_matrix_model(self):
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(self.y_test, self.y_predict)
        print("Confusion matrix: \n", confusion_matrix, "\n\n\n")
        return confusion_matrix

    # Precision, recall and "F1 score"
    def __report_model(self):
        from sklearn.metrics import classification_report
        classification_report = classification_report(self.y_test, self.y_predict)
        print("Classification report: \n", classification_report, "\n")
        return classification_report

    # 允许类的实例像函数一样被调用
    def __call__(self, *args, **kwargs):
        # Load data
        self.__load_iris_data()

        # Show data
        self.__show_iris_data()

        # Describe data
        self.__describe_iris_data()

        # Split data
        self.__split_iris_data()

        # Fit data
        self.__fit_KNC_model()

        # predict data
        self.__predict_KNC_model()

        # Accuracy
        self.__accuracy_KNC_model()

        # Confusion matrix
        self.__confusion_matrix_model()

        # Precision, recall and "F1 score"
        self.__report_model()


"""
    返回当前范围内的变量、方法和定义的类型列表
    dir(KNC_Classifier())
"""

if __name__ == "__main__":
    KNC()()
