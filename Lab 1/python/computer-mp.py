from matplotlib.pyplot import axis
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from multiprocessing import Pool 
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib as matplt
import matplotlib.pyplot as plt
import seaborn as sns
import time
import utils

serial = __import__('computer-serial')

## Part two – Parallel implementation, multiprocessing

## 1.- Write a parallel version of you program using multiprocessing

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


def elbow_graph_mp(X, total_k, seed_value):
    (n, p) = X.shape
    sum_sq_dist_total = np.zeros(total_k)
    num_cores = int(utils.available_cpu_count()/2)
    pool = mp.Pool(num_cores)
    res_X = np.zeros((n,p,total_k))
    try:
        kmeans_res = [pool.apply(serial.custom_kmeans, args=(X, k, seed_value)) for k in range(1, total_k+1)]
        for i in range(1, total_k+1):
            res_X = kmeans_res[:,:,i]
            sum_sq_dist = 0
            for j in range(i):
                elements_cluster = X[np.where(res_X[:,p]==j),]
                centroidekth = np.mean(X[np.where(res_X[:,p]==j),], axis=1)
                distance_centroid = np.sum(np.square(elements_cluster-centroidekth), axis=0)
                sum_sq_dist += np.sum(distance_centroid)
            sum_sq_dist_total[i-1] = sum_sq_dist
        pool.close() 
    except ValueError as err:
        print(err)
    return sum_sq_dist_total


if __name__== "__main__":
    df = pd.read_csv('computers_perform.csv')
    df_without_cat = df.drop(['id', 'laptop', 'cd', 'trend'], axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_without_cat)

    (n, p) = scaled_data.shape
    total_k = 10
    seed_value = 1234

    print(utils.available_cpu_count())

    start_time = time.time()
    elbow_graph_mp(scaled_data, total_k, seed_value)
    print('--- %s seconds ---' % (time.time() - start_time))


