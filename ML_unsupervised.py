# import Iris dataset

from sklearn import datasets

X, y = datasets.load_iris(return_X_y = True)
n_clusters = 3

# =============================================================================
# K-Means Clustering
# =============================================================================

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters = n_clusters)
k_means.fit(X)
print(k_means.labels_)

from sklearn.metrics import rand_score, jaccard_score, silhouette_score
print("Rand Index: {:.4f}".format(rand_score(y, k_means.labels_)))
print("Jaccard Score: {:.4f}".format(jaccard_score(y, k_means.labels_, average = 'macro')))
print("Silhouette Score: {:.4f}".format(silhouette_score(X, k_means.labels_)))

import matplotlib.pyplot as plt

fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d', elev = 48, azim = 134)
ax.set_position([0, 0, 0.95, 1])
labels = k_means.labels_
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c = labels.astype(float), edgecolor = 'k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.set_title("K-Means Clustering Result with n_clusters = {}".format(n_clusters))
ax.dist = 12

# =============================================================================
# Agglomerative Hierarchical Clustering
# =============================================================================

from sklearn.cluster import AgglomerativeClustering

agg_clustering = AgglomerativeClustering(n_clusters = n_clusters)
agg_clustering.fit(X)
print(agg_clustering.labels_)

print("Rand Index: {:.4f}".format(rand_score(y, agg_clustering.labels_)))
print("Jaccard Score: {:.4f}".format(jaccard_score(y, agg_clustering.labels_, average = 'macro')))
print("Silhouette Score: {:.4f}".format(silhouette_score(X, agg_clustering.labels_)))

agg_clustering2 = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'manhattan', linkage = 'average')
agg_clustering2.fit(X)
print(agg_clustering2.labels_)

print("Rand Index: {:.4f}".format(rand_score(y, agg_clustering2.labels_)))
print("Jaccard Score: {:.4f}".format(jaccard_score(y, agg_clustering2.labels_, average = 'macro')))
print("Silhouette Score: {:.4f}".format(silhouette_score(X, agg_clustering2.labels_)))
