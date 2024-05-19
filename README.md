### Clustering
This is a repository for Clustering Algorithm in Machine Learning, including K-means, DBSCAN, Spectral clustering, Hierarchical Clustering and a video demo based on K_means.

### K-means Clustering Basic Steps:

1. Select K initial centroids.
2. Assign each data point to the nearest centroid, forming K clusters.
3. Compute the centroid of each cluster.
4. Repeat steps 2 and 3 until the centroids no longer change or the maximum number of iterations is reached.

### DBSCAN Clustering Basic Steps:

1. For each point, if the number of points within its neighborhood is greater than or equal to MinPts, mark it as a core point.
2. For each core point, expand all directly density-reachable points to form a cluster.
3. Repeat step 2 until all core points have been visited.
4. Mark all non-core points as noise points.

### Hierarchical Clustering Basic Steps:

1. Treat each data point as a single cluster.
2. Calculate the distance between all clusters and merge the two closest clusters.
3. Repeat step 2 until all points are merged into a single cluster or the desired number of clusters is reached.

### Spectral Clustering Basic Steps:

1. Construct a similarity matrix (usually using k-nearest neighbors).
2. Compute the eigenvalues and eigenvectors of the Laplacian matrix.
3. Select the top k eigenvectors to form a new representation space.
4. In the new representation space, use K-means clustering to form clusters.

These steps outline the fundamental logic and processes involved in each clustering algorithm.
