import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class unsupervisedFuncs:
    def __init__(self, data_in):
        self.data = data_in
        self.projected_data = []
        self.y_data = []
        self.y_prob = []
        #self.kmc_centers = []

    def let_PCA(self, components=2):
        pca = PCA(n_components=components)
        projected = pca.fit_transform(self.data)
        print("original data shape: ", self.data.shape)
        print("transformed data shape:  ", projected.shape)
        self.projected_data = projected

    def let_maniford(self, method=0, components=2):
        if method == 0: # MDS
            model = MDS(n_components=components, random_state=2)

        elif method == 1: # LLE
            model = LocallyLinearEmbedding(n_neighbors=5, n_components=components)

        elif method == 2: # Isomap
            model = Isomap(n_components=components) 
        
        else: # TSNE
            model = TSNE(n_components=components)

        out = model.fit_transform(self.data)
        self.projected_data = out

    def let_kMC(self, clusters=10):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(self.projected_data)
        y_kmeans = kmeans.predict(self.projected_data)
        #self.kmc_centers = kmeans.cluster_centers_
        self.y_data = y_kmeans

    def let_GMM(self, clusters=10):
        gmm = GaussianMixture(n_components=clusters).fit(self.projected_data)
        y_gmm = gmm.predict(self.projected_data)
        y_prob = gmm.predict_proba(self.projected_data)
        self.y_prob = y_prob
        self.y_data = y_gmm

    def show_components_info(self):
        pca = PCA().fit(self.data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def print_plot(self):
        fig = plt.figure(1)
        #print
        if self.projected_data.shape[1] == 2:
            plt.subplot(121)
            plt.scatter(self.projected_data[:,0], self.projected_data[:,1], alpha=0.5)
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.subplot(122)
            plt.scatter(self.projected_data[:,0], self.projected_data[:,1], c=self.y_data, s=30, cmap='viridis', alpha=0.8) 
            #plt.scatter(self.kmc_centers[:,0], self.kmc_centers[:,1], c='black', s=200, alpha=0.5)
            plt.xlabel('component 1')
            plt.ylabel('component 2')
        
        elif self.projected_data.shape[1] == 3:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(self.projected_data[:,0], self.projected_data[:,1], self.projected_data[:,2], alpha=0.5)
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(self.projected_data[:,0], self.projected_data[:,1], self.projected_data[:,2], c=self.y_data, s=30, cmap='viridis', alpha=0.8)
            #ax.scatter(self.kmc_centers[:,0], self.kmc_centers[:,1], self.kmc_centers[:,2], c='black', s=200, alpha=0.5) 
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')

        else:
            return

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
