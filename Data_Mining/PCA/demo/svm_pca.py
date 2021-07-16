# encoding=utf-8

# Time : 2021/5/30 14:00 

# Author : 啵啵

# File : svm_pca.py 

# Software: PyCharm
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report,accuracy_score

class Svm_pca(object):

    def __init__(self):
        self.lfw_people = None
        self.images = None
        self.target_names = None
        self.target = None
        self.data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_pca = None
        self.x_test_pca = None

    def __load_data(self):
        self.lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    def __detailed_information(self):
        # ['data', 'images', 'target', 'target_names', 'DESCR']
        self.data = self.lfw_people.get('data')
        self.images = self.lfw_people.get('images')
        n_samples = self.images.shape[0]
        self.target = self.lfw_people.get('target')
        n_features = self.lfw_people.get('data').shape[1]
        self.target_names = self.lfw_people.get('target_names')
        n_classes = self.lfw_people.get('target_names').shape[0]
        print(f"n_samples:{n_samples}\n"
              f"n_features:{n_features}\n"
              f"target_names:{self.target_names}\n"
              f"n_classes:{n_classes}")

    def __plot_images(self, images,pic_names,titles):
        plt.figure(figsize=(12, 10))
        for index in range(12):
            plt.subplot(3, 4, index + 1)
            plt.imshow(images[index], cmap=plt.cm.gray)
            plt.title(f"true label: {titles[index]}")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(pic_names,bbox_inches='tight',dpi=300)

    def __cov_cal(self):
        eigen_values, eigen_vectors = np.linalg.eig(np.cov(self.data))
        return eigen_values, eigen_vectors

    def __train_test_split(self):
        x = self.lfw_people.get('data')
        y = self.lfw_people.get('target')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=66, test_size=0.25)

    def __pca_model(self):
        pca = PCA(n_components=150, svd_solver="randomized", whiten=True)
        pca.fit(self.x_train)
        self.engenfances = pca.components_.reshape((150, self.images.shape[1], self.images.shape[2]))
        self.x_train_pca = pca.transform(self.x_train)
        self.x_test_pca = pca.transform(self.x_test)
        print(f"explained_variance_ratio_sum:{pca.explained_variance_ratio_.sum()}")

    def __explore_pca(self):
        plt.figure(figsize=(8,6))
        superpa = []
        for n_components in range(150, min(self.x_train.shape), 50):
            pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
            pca.fit(self.x_train)
            superpa.append(pca.explained_variance_ratio_.sum())
        print(superpa)
        plt.plot(range(150, min(self.x_train.shape), 50), superpa)
        plt.savefig("explore_pca_pic.png", bbox_inches='tight', dpi=300)

    def __svm_model(self):
        # GridSearch -> C and -> gamma
        svc = SVC(kernel='rbf', class_weight='balanced',C=1e3,gamma=0.005)
        svc.fit(self.x_train_pca,self.y_train)
        y_pred = svc.predict(self.x_test_pca)

        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        print(accuracy_score(self.y_test, y_pred))

    def __call__(self, *args, **kwargs):
        self.__load_data()
        self.__detailed_information()
        self.__plot_images(self.images,pic_names="samples.png",titles=[self.target_names[self.target[index]] for index in range(12)])
        self.__train_test_split()
        self.__pca_model()
        self.__plot_images(self.engenfances,pic_names="engenfances.png",titles=[f"engenfance:{index}" for index in range(12)])
        self.__explore_pca()
        self.__svm_model()

if __name__ == '__main__':
    Svm_pca()()
