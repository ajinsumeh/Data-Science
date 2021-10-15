from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

"""
centroidLocation: Coordinates of the centroids that will generate the random data.

Example: input: [[4,3], [2,-1], [-1,4]]
numSamples: The number of data points we want generated, split over the number of centroids (# of centroids defined in centroidLocation)

Example: 1500
clusterDeviation: The standard deviation of the clusters. The larger the number, the further the spacing of the data points within the clusters.

Example: 0.5
"""

num_samples_total = 1000
cluster_centers = [(3,3), (7,7)]
num_classes = len(cluster_centers)
epsilon = 1

"""
DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.

It works based on two parameters: Epsilon and Minimum Points Epsilon determine a specified radius that if includes enough number of points within, we call it dense area minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.
"""

min_samples = 13

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)

db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
labels = db.labels_
labels

no_clusters = len(np.unique(labels) ) # output is 2
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)
no_noise


colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#aeb404', labels)) # the codes represent the hex color code
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title('Two clusters with data')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()
