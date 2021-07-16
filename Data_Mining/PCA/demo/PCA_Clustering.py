# encoding=utf-8

# Time : 2021/6/3 13:59 

# Author : 啵啵

# File : PCA_Clustering.py 

# Software: PyCharm

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA

columns = pd.read_csv("../data/segmentation_data/segmentation_names.txt", header=None).values.reshape(1, -1)[0]
labels = pd.read_csv("../data/segmentation_data/segmentation_classes.txt", sep='\t', header=None)
labels.columns = ['label_names', 'label_numbers']
features = pd.read_csv("../data/segmentation_data/segmentation_data.txt", header=None, )
features.columns = columns

data = pd.concat([features, labels], axis=1).drop(['label_names'], axis=1)
X = data.drop(['label_numbers'], axis=1)
y = data['label_numbers']
scale_X = MinMaxScaler(feature_range=[0, 1]).fit_transform(X)

def kmeans_model(X):
    Sum_of_squared_distances = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k).fit(X)
        Sum_of_squared_distances.append(kmeans.inertia_)

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 15), Sum_of_squared_distances, 'rx-')
    plt.xlabel('$K$')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('$Elbow\ Method\ For\ Optimal\ K$')
    plt.savefig("kmeans_original.png")

def rfc_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66, shuffle=True)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pre = rfc.predict(X_test)
    print(accuracy_score(y_test, y_pre))
    print(classification_report(y_test, y_pre))
    print("\n")

K = 7
# plt.show()
rfc_model(scale_X, y)

chi2_values = chi2(scale_X, y)[0]
# print(chi2_values)

scale_X_chi2 = SelectKBest(chi2, k=10).fit_transform(scale_X, y)

rfc_model(scale_X_chi2, y)

def pca_model(X):
    plt.figure(figsize=(12, 8))
    superpa = []
    for n_components in range(1, min(X.shape)):
        pca = PCA(n_components=n_components, svd_solver="full", whiten=True)
        pca.fit(scale_X)
    plt.plot(range(1, min(X.shape)), superpa)
    plt.savefig("explore_pca_pic.png", bbox_inches='tight', dpi=300)
pca_model(scale_X)

n_components=7

scale_X_pca = PCA(n_components=n_components).fit_transform(scale_X)

kmeans_model(scale_X_pca)

rfc_model(scale_X_pca,y)