import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
outputfile = 'LDA6.xlsx' #降维后的数据

def lda(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''

    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #within_class scatter matrix
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #between_class scatter matrix
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw)*SB
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim

if __name__ == '__main__':
    data1 = pd.read_excel('yy.xlsx',engine="openpyxl", names=range(0, 129))
    X = data1.drop([0], axis=1)
    data = data1.values
    Y = data[:, 0]
    #
    # iris = load_iris()
    # X = iris.data
    # Y = iris.target
    data_1 = lda(X, Y, 5)

    data_2 = LinearDiscriminantAnalysis(n_components=5).fit_transform(X, Y)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("LDA")
    # plt.xlim([data_1[:, 0].min()*1.2,data_1[:, 0].max()*1.2])
    # plt.ylim([data_1[:, 1].min()*1.3,data_1[:, 1].max()*1.3])
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("LDA4.png",dpi=600)
    plt.show()

    writer=pd.ExcelWriter(outputfile)
    pd.DataFrame(data_1).to_excel(writer,sheet_name='LDA',)
    pd.DataFrame(data_2).to_excel(writer,sheet_name='sklearn_LDA')
    # pd.DataFrame(feature_vector).to_excel(writer,sheet_name='特征向量')
    # pd.DataFrame(scale).to_excel(writer,sheet_name='标准化数据')

    writer.save()
