# encoding=utf-8

# Time : 2021/5/20 15:06 

# Author : 啵啵

# File : Clustering_Pro2.py 

# Software: PyCharm

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA

fpath = r'../data/Wholesale customers data.csv'
data = pd.read_csv(fpath, sep=",")
categorical_features = data[['Channel', 'Region']]
continuous_features = data.drop(['Channel', 'Region'],axis=1)
dummies = OneHotEncoder(categories="auto").fit_transform(categorical_features).toarray()
new_data = pd.concat([pd.DataFrame(dummies),continuous_features],axis = 1)
scale_data = MinMaxScaler(feature_range=[0,1]).fit_transform(new_data)

Sum_of_squared_distances = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k).fit(scale_data)
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(12,8))
plt.plot(range(1,15),Sum_of_squared_distances,'bx-')
plt.xlabel('$K$')
plt.ylabel('Sum_of_squared_distances')
plt.title('$Elbow\ Method\ For\ Optimal\ K$')
plt.show()

K = 6

labels = AgglomerativeClustering(n_clusters=K).fit_predict(scale_data)

pca = PCA(n_components=2).fit(scale_data)
nd = pca.transform(scale_data)
print(nd)
print(labels)
plt.figure(figsize=(12,8))
colors = ['red','green','yellow','blue','black','orange',]
for num in range(K):
    plt.scatter(nd[labels==num,0], nd[labels==num,1],label=str(num),color=colors[num],linewidths=10)
plt.legend()
plt.savefig("scatter.png",bbox_inches='tight',dpi=300)
plt.show()

########################
plt.figure(figsize=(12,8))
sch.dendrogram(sch.linkage(scale_data, method='ward'))
plt.plot(range(10000),[17.5]*10000,color='r',lw=5,ls='--')
plt.xticks([])
plt.yticks([])
plt.title("$Dendrogram$")
plt.show()