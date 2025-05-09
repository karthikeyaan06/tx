#pip install numpy scikit-learn scipy
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
X = np.random.rand(10, 2)  # Sample dataset with 10 points in 2D
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
centroids = kmeans.cluster_centers_
def bregman_divergence(x, c):
    return distance.euclidean(x, c) ** 2
for i, x in enumerate(X):
    print(f"Point {i} Bregman divergence: {bregman_divergence(x, centroids[0])}")
