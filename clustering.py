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

from yellowbrick.cluster import SilhouetteVisualizer

#from clusteval import clusteval

#--------------------------------------------------------------------------------------


def make_df(graph):
    df = pd.DataFrame(list(graph.items()))

    df.rename(columns = {0:'word', 1:'vector'}, inplace = True)
    df[[i for i in range(0, 50)]] = pd.DataFrame(df['vector'].tolist(), index=df.index)
    df.drop('vector', axis=1, inplace=True)

    df = df.drop('word', axis=1)

    return df


def apply_pca(df):
    pca = PCA(n_components=2).fit_transform(df)

    #pcaratio = pca.explained_variance_ratio_
    return pca


def clustevalres(X):
    ce = clusteval(evaluate='silhouette')
    ce.fit(X)
    #ce.plot()
    #ce.dendrogram()
    ce.scatter(X)

    ce = clusteval(evaluate='dbindex')
    ce.fit(X)
    #ce.plot()
    ce.scatter(X)
    #ce.dendrogram()

    ce = clusteval(cluster='dbscan')
    try:
        ce.fit(X)
        ce.plot()
        ce.scatter(X)
    except ValueError:
        pass
    #ce.dendrogram()

    ce = clusteval(cluster='hdbscan')
    ce.fit(X)
    #ce.plot()
    ce.scatter(X)
    #ce.dendrogram()


def vary_damping(graph, algo):

    for i in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print("damping:", i)
        try:
            algo(graph, damping=i)
        except ValueError:
            print("Damping", i, " resulted in just one big cluster")

        print("""---""")

def vary_n_of_clusters(graph, algo):
    for n in range(2, 15):
        print("n:", n)
        algo(graph, n)
        print("""---""")

def vary_min_samples(graph, algo):
    for n in range(4, 15):
        print("n:", n)
        algo(graph, n)
        print("""---""")


def scatter_plot_aff(model, X):
    #model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    from matplotlib.cm import get_cmap
    colors = get_cmap("tab10").colors + get_cmap("Dark2").colors + get_cmap("tab10").colors  # type: list
    i = 0
    
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    
    #pyplot.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        try: pyplot.scatter(X[row_ix, 0], X[row_ix, 1], c=colors[i])#.reshape(1,-1)) #, linewidth=1, alpha=0.5, c=np.random.rand(len(clusters))/255)
        except IndexError: break
        i += 1
    #pyplot.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    pyplot.grid(False)    
    pyplot.show()

    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    print("Sillhouette score: ", sil_score)
    print("Percentage score: ", (sil_score+1)/2)
    print("Number of clusters: ", len(model.cluster_centers_indices_))

def scatter_plot_rest(model, X, n):
    model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    from matplotlib.cm import get_cmap
    colors = get_cmap("tab10").colors + get_cmap("Dark2").colors + get_cmap("tab10").colors  # type: list
    #colors = get_cmap("Set3").colors + get_cmap("Pastel1").colors + get_cmap("Pastel2").colors  # type: list
    i = 0
    
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        try: pyplot.scatter(X[row_ix, 0], X[row_ix, 1], c=colors[i])#.reshape(1,-1))       #, linewidth=1, alpha=0.5, c=np.random.rand(len(clusters))/255)
        except IndexError: break
        i += 1
        #print(colors)
        #print(i)
        #print(colors[i])
    #pyplot.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    pyplot.grid(False)    
    pyplot.show()

    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    print("Sillhouette score: ", sil_score)
    print("Percentage score: ", (sil_score+1)/2)
    print("Number of clusters: ", n)


def cluster_with_affinity_propagation(X, damping):
    from sklearn.cluster import AffinityPropagation
    model = AffinityPropagation(damping=damping)
    scatter_plot_aff(model, X)
    
    
def agglomerative_clustering(X, n):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n)
    scatter_plot_rest(model, X, n)
    
'''def agglomerative_clustering(X, n):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])#, s=3)
    pyplot.show()
    labels = model.labels_
    sil_score = metrics.silhouette_score(X, labels, metric="sqeuclidean")

    print("Sillhouette score: ", sil_score)
    print("Percentage score: ", (sil_score+1)/2)
    print("Number of clusters: ", n)'''


def cluster_with_birch(X, n):
    from sklearn.cluster import Birch
    model = Birch(threshold=0.01, n_clusters=n)
    model.fit(X)
    scatter_plot_rest(model, X, n)


def cluster_with_kmeans(X, n):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n)
    model.fit(X)
    scatter_plot_rest(model, X, n)


def cluster_with_mini_batch_kmeans(X, n):
    from sklearn.cluster import MiniBatchKMeans
    model = MiniBatchKMeans(n_clusters=n)
    model.fit(X)
    yhat = model.predict(X)
    scatter_plot_rest(model, X, n)


def spectral_clustering(X, n):
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(n_clusters=n)
    scatter_plot_rest(model, X, n)

def mean_shift_clustering(X):
    from sklearn.cluster import MeanShift
    model = MeanShift()
    scatter_plot_rest(model, X, n)

def cluster_with_optics(X, n):
    from sklearn.cluster import OPTICS
    model = OPTICS(eps=0.8, min_samples=n)
    scatter_plot_rest(model, X, n)

def cluster_with_dbscan(X, n):
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=0.30, min_samples=n)
    scatter_plot_rest(model, X, n)


def plot_with_annotations(graph):
    import numpy as np
    import matplotlib.pyplot as plt

    df = apply_pca(make_df(graph))
    df = list(df)
    x = [i[0] for i in df]
    y = [i[1] for i in df]
    graph = sa_la_graph
    df = pd.DataFrame(list(graph.items()))
    df.rename(columns = {0:'word', 1:'vector'}, inplace = True)
    annotations = list(df['word'])

    plt.figure(figsize=(30,15))
    plt.scatter(x, y,s=100,color="red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with annotations",fontsize=15)
    for i, label in enumerate(annotations):
        plt.annotate(label, (x[i], y[i]))

    plt.show()


def plot_clusters_old(df):
    """not in use"""
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
