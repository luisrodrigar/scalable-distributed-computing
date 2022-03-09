from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("computers_dev.csv")
df_without_cat = df.drop(["id", "laptop", "cd", "trend"], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_without_cat)

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
            distance_cluster[:,i] = np.sqrt(np.sum(np.square(np.subtract(X, centroids[i,:])), axis=0))

        cluster = np.zeros(n)
        for i in range(n):
            cluster[i] = np.where(distance_cluster[i,] == np.min(distance_cluster[i, ]))[0][0]
        assign_cluster = np.concatenate([X,cluster], axis=1)

        new_centroids = np.zeros((k, p))
        for i in range(k):
            x_index_kth_cluster = np.where(assign_cluster[:,p+1] == i)[0][0]
            x_kth_cluster = X[x_index_kth_cluster]
            kth_centroid = np.mean(x_kth_cluster, axis=1)
            new_centroids[i,] = kth_centroid
        if (new_centroids==centroids).all():
            centroids_not_equal = False
        else:
            centroids = new_centroids
        ite += 1
    return assign_cluster


res = custom_kmeans(scaled_data, 2, 12345)
