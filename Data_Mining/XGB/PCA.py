import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


# def pca(X, k):  # k is the components you want
#     # mean of each feature
#     n_samples, n_features = X.shape
#     mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
#     # normalization
#     norm_X = X - mean
#     # scatter matrix
#     scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
#     # Calculate the eigenvectors and eigenvalues
#     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs.sort(reverse=True)
#     # select the top k eig_vec
#     feature = np.array([ele[1] for ele in eig_pairs[:k]])
#     # get new data
#     data = np.dot(norm_X, np.transpose(feature))
#     return data


# inputfile = 'wy.csv'#原始数据
outputfile = r'./28PCA6.xlsx'  # 降维后的数据

data1 = pd.read_excel('./yy.xlsx', engine="xlrd",names=range(0, 129))
# X = data1.drop([0], axis=1)
#
# iris = load_iris()
# X = iris.data
# Y = iris.target
# data_2 = LinearDiscriminantAnalysis(n_components=5).fit_transform(X, Y)

# data1 = pd.read_excel(r'../DATA/yy.xlsx', engine='open yyxlsx', names=range(129))  # 读入数据

# 特征列
data = data1.drop([0], axis=1)

# 标准化数据
scale = (data - data.mean()) / (data.std())

# 查看数据是否正常

X = data.values
Y = data1.values[:, 0]
from sklearn.decomposition import PCA

# 保留所有成分
pca = PCA(n_components=5)
pca.fit(scale)

data_1 = pca.transform(scale)

# 降维前
plt.figure(figsize=(4, 4))
# plt.subplot(121)
plt.title("PCA")
plt.scatter(data_1[:, 0], data_1[:, 1], c=Y)
plt.savefig("pca.png",dpi=300)
# plt.show()


# 返回模型的各个特征向量
feature_vector = pca.components_

# 返回各个成分各自的方差百分比(也称贡献率）
contri_rate = pca.explained_variance_ratio_
print(contri_rate.sum())
# 查看贡献率
# print(contri_rate.shape)

# 选取累计贡献率大于80%的主成分（3个主成分）
# pca = PCA(6)
# pca.fit(scale)
# # 降低维度
# low_d = pca.transform(scale)
# newx = pca.inverse_transform(low_d)
# print(newx)
# print(newx.shape)

# 将结果写入excel
writer=pd.ExcelWriter(outputfile)
pd.DataFrame(data_1).to_excel(writer,sheet_name='主成分')
pd.DataFrame(contri_rate).to_excel(writer,sheet_name='贡献率')
pd.DataFrame(feature_vector).to_excel(writer,sheet_name='特征向量')
pd.DataFrame(scale).to_excel(writer,sheet_name='标准化数据')
writer.save()

# 降维后
# plt.subplot(122)
# plt.title("A_PCA")
# plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)
# plt.savefig("LDA4.png", dpi=600)


