from matplotlib.pyplot import axis
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib as matplt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
import time

df = pd.read_csv('computers_dev.csv')
df_without_cat = df.drop(['id', 'laptop', 'cd', 'trend'], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_without_cat)

(n, p) = scaled_data.shape

# Part one – Serial version

## 1.- Construct the elbow graph and find the optimal clusters number (k).
## OPTION A

## 2.- Implement the k-means algorithm

def custom_kmeans(X, k, seed_value):
    (n, p) = X.shape
    assign_cluster = np.zeros((n, p+1))
    centroids_not_equal = True
    ite = 0
    np.random.seed(seed_value)
    centroids_index = np.random.choice(range(n), k)
    centroids = X[centroids_index, :]
    while(centroids_not_equal):
        distance_cluster = np.zeros((n, k))
        for i in range(k):
            distance_cluster[:,i] = np.sqrt(np.sum(np.square(np.subtract(X, np.atleast_2d(centroids[i,:]))), axis=1))
        cluster = np.zeros(n)
        for i in range(n):
            cluster[i] = np.where(distance_cluster[i,] == np.min(distance_cluster[i, ]))[0][0]
        assign_cluster = np.append(X, np.atleast_2d(cluster).T, axis=1)
        new_centroids = np.zeros((k, p))
        for i in range(k):
            x_index_kth_cluster = np.where(assign_cluster[:,p] == i)[0]
            x_kth_cluster = X[x_index_kth_cluster,]
            kth_centroid = np.mean(x_kth_cluster, axis=0)
            new_centroids[i,] = kth_centroid
        if (new_centroids==centroids).all():
            centroids_not_equal = False
        else:
            centroids = new_centroids
        ite += 1
    return assign_cluster

## 3.- Cluster the data using the optimum value using k-means.

def elbow_graph(X, total_k, seed_value, print_graph=False):
    (n, p) = X.shape
    sum_sq_dist_total = np.zeros(total_k)
    for i in range(1, total_k+1):
        res_X = custom_kmeans(X, i, seed_value)
        sum_sq_dist = 0
        for j in range(i):
            elements_cluster = X[np.where(res_X[:,p]==j),]
            centroidekth = np.mean(X[np.where(res_X[:,p]==j),], axis=1)
            distance_centroid = np.sum(np.square(elements_cluster-centroidekth), axis=0)
            sum_sq_dist += np.sum(distance_centroid)
        sum_sq_dist_total[i-1] = sum_sq_dist
    if print_graph:
        plt.plot(np.array(range(10))+1, sum_sq_dist_total, '-ob')
        plt.xlabel('Number of clusters')
        plt.ylabel('Total Sum of Squares')
        plt.title('Elbow graph')
        plt.show()
    return sum_sq_dist_total


elbow_graph(scaled_data, 10, 1234)

optiomal_k = 2
res = custom_kmeans(scaled_data, optiomal_k, 12345)

## 4. - Measure time

start_time = time.time()
elbow_graph(scaled_data, 10, 1234)
print('--- %s seconds ---' % (time.time() - start_time))

start_time = time.time()
custom_kmeans(scaled_data, optiomal_k, 12345)
print('--- %s seconds ---' % (time.time() - start_time))

## 5. - Plot the results of the elbow graph.

elbow_graph(scaled_data, 10, 1234, True)

## 6. - Plot the first 2 dimensions of the clusters
scatter = plt.scatter(scaled_data[:,0], scaled_data[:,1], cmap=plt.get_cmap('viridis'), c=res[:,p], label=res[:,p])
plt.xlabel('price')
plt.ylabel('speed')
plt.title('Scatter Plot of two first dimensions with 2 clusters')
plt.legend(*scatter.legend_elements(), loc='upper left', title='Clusters')
plt.show()

## 7. - Find the cluster with the highest average price and print it.

res_df = pd.DataFrame(res, columns=np.append(df_without_cat.columns, 'cluster'))

res_df_group = res_df.groupby('cluster').mean('price')
cluster_id = int(res_df_group['price'].idxmax())

print('The cluster with the highest average price is {}'.format(cluster_id))

## 8. - Print a heat map using the values of the clusters centroids.

viridis_pal = cm.get_cmap('viridis', 2)
colors = [matplt.colors.rgb2hex(viridis_pal(int(item))) for item in res[:,p]]

row_linkage = hierarchy.linkage(scaled_data, method='centroid')
g = sns.clustermap(res_df.iloc[:,list(range(0,p))], cmap=colors, row_linkage=res_df.iloc[:,p], row_cluster=False)
plt.legend(*scatter.legend_elements(), loc='upper left', title='Clusters')
plt.show()

## Part two – Parallel implementation, multiprocessing
