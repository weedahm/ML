import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def let_PCA(data, components=2):

    pca = PCA(n_components=components)
    projected = pca.fit_transform(data)
    print("original data shape: ", data.shape)
    print("transformed data shape:  ", projected.shape)
    x = projected[:,0]
    y = projected[:,1]
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

    ##### practice
    '''
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

    pca = PCA(n_components = 1)
    pca.fit(X)
    print(pca.components_)
    X_pca = pca.transform(X)
    print("original shape:  ", X.shape)
    print("transformed shape:   ", X_pca.shape)
    X_new = pca.inverse_transform(X_pca)
    print(X_new.shape)

    #plt.plot(X[:,0], X[:,1])
    x = X[:,0]
    y = X[:,1]
    x_new = X_new[:,0]
    y_new = X_new[:,1]
    #print(x.shape, y.shape)
    plt.scatter(x, y, alpha=0.2)
    plt.scatter(x_new, y_new, alpha=0.8)
    plt.show()
    '''

def show_components_info(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
