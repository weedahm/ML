import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class unsupervised_funcs:
    def __init__(self, data_in):
        self.data = data_in
        self.projected_data = []
        self.y_kMC_data = []
        self.kmc_centers = []

    def let_PCA(self, components=2):
        pca = PCA(n_components=components)
        projected = pca.fit_transform(self.data)
        print("original data shape: ", self.data.shape)
        print("transformed data shape:  ", projected.shape)
        self.projected_data = projected

    def show_components_info(self):
        pca = PCA().fit(self.data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def let_kMC(self, clusters=10):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(self.projected_data)
        y_kmeans = kmeans.predict(self.projected_data)
        self.kmc_centers = kmeans.cluster_centers_
        self.y_kMC_data = y_kmeans

    def print_plot(self):
        fig = plt.figure(1)
        #print
        if self.projected_data.shape[1] == 2:
            plt.subplot(121)
            plt.scatter(self.projected_data[:,0], self.projected_data[:,1], alpha=0.5)
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.subplot(122)
            plt.scatter(self.projected_data[:,0], self.projected_data[:,1], c=self.y_kMC_data, s=30, cmap='viridis', alpha=0.8) 
            plt.scatter(self.kmc_centers[:,0], self.kmc_centers[:,1], c='black', s=200, alpha=0.5)
            plt.xlabel('component 1')
            plt.ylabel('component 2')
        
        elif self.projected_data.shape[1] == 3:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(self.projected_data[:,0], self.projected_data[:,1], self.projected_data[:,2], alpha=0.5)
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(self.projected_data[:,0], self.projected_data[:,1], self.projected_data[:,2], c=self.y_kMC_data, s=30, cmap='viridis', alpha=0.8)
            ax.scatter(self.kmc_centers[:,0], self.kmc_centers[:,1], self.kmc_centers[:,2], c='black', s=200, alpha=0.5) 
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()