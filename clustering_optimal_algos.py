from numpy.lib import nanfunctions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import plotly_express as px
import plotly.graph_objs as go
#import plotly.plotly as py
from sklearn.decomposition import PCA

import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
import pickle
import time

from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn import metrics

#from clusteval import clusteval

#--------------------------------------------------------------------------------------


def cluster_with_affinity_propagation(X, damping):
    from sklearn.cluster import AffinityPropagation
    model = AffinityPropagation(damping=damping)
    model.fit(X)

    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def agglomerative_clustering(X, n):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def cluster_with_birch(X, n):
    from sklearn.cluster import Birch
    model = Birch(threshold=0.01, n_clusters=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def cluster_with_kmeans(X, n):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def cluster_with_mini_batch_kmeans(X, n):
    from sklearn.cluster import MiniBatchKMeans
    model = MiniBatchKMeans(n_clusters=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def spectral_clustering(X, n):
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(n_clusters=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    return sil_score, (sil_score+1)/2


def mean_shift_clustering(X):
    from sklearn.cluster import MeanShift
    model = MeanShift()
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def cluster_with_optics(X, n):
    from sklearn.cluster import OPTICS
    model = OPTICS(eps=0.8, min_samples=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    return sil_score, (sil_score+1)/2


def cluster_with_dbscan(X, n):
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=0.30, min_samples=n)
    model.fit(X)
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    return sil_score, (sil_score+1)/2


def plot_clusters(df):
    '''using tsne'''
    pca = PCA().fit(df)

    pcaratio = pca.explained_variance_ratio_
    trace = go.Scatter(x=np.arange(len(pcaratio)),y=np.cumsum(pcaratio))
    data = [trace]
    layout = dict(title="Results")
    fig = dict(data=data, layout=layout)

    pca = PCA(n_components=5)
    sPCA = pca.fit_transform(df)
    print("info retained: ", pca.explained_variance_ratio_)


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6)
    sPCA_labels = kmeans.fit_predict(sPCA)

    dfPCA = pd.DataFrame(sPCA)
    dfPCA['cluster'] = sPCA_labels


    from sklearn.manifold import TSNE
    X = dfPCA.iloc[:,:-1]
    Xtsne = TSNE(n_components=2).fit_transform(X)
    dftsne = pd.DataFrame(Xtsne)
    dftsne['cluster'] = sPCA_labels
    dftsne.columns = ['x1','x2','cluster']


    fig = plt.plot(figsize=(10, 6))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5)#,ax=ax[0])
    #fig.title('Hindi-German')
    #sns.scatterplot(data=dfsPCA2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[1])
    #ax[1].set_title('Visualized on PCA 2D')
    #fig.suptitle('Comparing clustering result when visualized using TSNE2D vs. PCA2D')
    display(fig)
