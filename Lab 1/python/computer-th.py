from matplotlib import cm
from _thread import allocate_lock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import time
import utils

# Part three – Parallel implementation, threading

# 1. - Write a parallel version of you program using threads

lock = allocate_lock()

def calc_new_centroids(X, assign_cluster, i, p, new_centroids):
    x_index_kth_cluster = np.where(assign_cluster[:,p] == i)[0]
    x_kth_cluster = X[x_index_kth_cluster,]
    kthcentroid =  np.mean(x_kth_cluster, axis=0)
    lock.acquire()
    new_centroids.append(kthcentroid)
    lock.release()
    return kthcentroid


def calc_distance_centroids(X, centroids, i, distances):  
    distance = np.sqrt(np.sum(np.square(np.subtract(X, np.atleast_2d(centroids[i,:]))), axis=1))
    lock.acquire()
    distances.append(distance)
    lock.release()
    return distance


def calc_cluster_associations(distance_cluster, i, clusters):  
    cluster = np.where(distance_cluster[i,] == np.min(distance_cluster[i, ]))[0][0]
    lock.acquire()
    clusters.append(cluster)
    lock.release()
    return cluster


def custom_kmeans_th(X, k, seed_value):

    (n, p) = X.shape
    assign_cluster = np.zeros((n, p+1))
    centroids_not_equal = True
    ite = 0
    np.random.seed(seed_value)
    centroids_index = np.random.choice(range(n), k)
    centroids = X[centroids_index, :]

    while(centroids_not_equal):
        th_distances, th_cluster_associated, th_centroids = [], [], []
        distances, clusters, new_centroids = list(), list(), list()
        [th_distances.append(threading.Thread(target=calc_distance_centroids, args=(X, centroids, i, distances))) for i in range(k)]
        [th_distances[i].start() for i in range(k)]
        [th_distances[j].join() for j in range(k)]
        distances = np.array(distances).T
        [th_cluster_associated.append(threading.Thread(target=calc_cluster_associations, args=(distances, i, clusters))) for i in range(n)]
        [th_cluster_associated[i].start() for i in range(n)]
        [th_cluster_associated[j].join() for j in range(n)]
        assign_cluster = np.append(X, np.atleast_2d(clusters).T, axis=1)
        [th_centroids.append(threading.Thread(target=calc_new_centroids, args=(X, assign_cluster, i, p, new_centroids))) for i in range(k)]
        [th_centroids[i].start() for i in range(k)]
        [th_centroids[j].join() for j in range(k)]
            
        if (new_centroids==centroids).all():
            centroids_not_equal = False
        else:
            centroids = np.array(new_centroids)
        ite += 1
    return assign_cluster

def custom_kmeans(X, k, seed_value, kmeans_res):
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
    kmeans_res.append(assign_cluster)
    return assign_cluster


def calc_sum_sq_distances(kmeans_res, i, sum_sq_dist_total):
    index_k_centroids = i-1
    res_X = kmeans_res[index_k_centroids]
    (_, p) = res_X.shape
    sum_sq_dist = 0
    for j in range(i):
        elements_cluster = res_X[np.where(res_X[:,p-1]==j),]
        centroidekth = np.mean(res_X[np.where(res_X[:,p-1]==j),], axis=1)
        distance_centroid = np.sum(np.square(elements_cluster-centroidekth), axis=0)
        sum_sq_dist += np.sum(distance_centroid)
    sum_sq_dist_total.append(sum_sq_dist)
    return sum_sq_dist


def elbow_graph_th(data, total_k, seed_value):
    th_kmeans = []
    kmeans_res, sum_sq_dist_total = list(), list()
    [th_kmeans.append(threading.Thread(target=custom_kmeans, args=(data, i, seed_value, kmeans_res))) for i in range(1, total_k+1)]
    [th_kmeans[i].start() for i in range(total_k)]
    [th_kmeans[j].join() for j in range(total_k)]
    th_sum_square = []
    [th_sum_square.append(threading.Thread(target=calc_sum_sq_distances, args=(kmeans_res, i, sum_sq_dist_total))) for i in range(1, total_k+1)]
    [th_sum_square[j].start() for j in range(total_k)]
    [th_sum_square[j].join() for j in range(total_k)]
    return sum_sq_dist_total

if __name__== "__main__":
    df = utils.perform_dataset()
    df_without_cat = utils.tiny_data(df)
    scaled_data = utils.scale_data(df_without_cat)

    (n, p) = scaled_data.shape
    total_k = 3
    optimal_k = 2
    seed_value = 1234

    # 2. - Measure the time and optimize the program to get the fastest version you can.

    ####################################
    # Measure the time for the k-means #
    ####################################

    ## Call one the function custom_kmeans once and check the time consumption
    start_time = time.time()
    res = custom_kmeans_th(scaled_data, optimal_k, seed_value)
    print('--- %s seconds ---' % (time.time() - start_time))
    ## --- 125.03917098045349 seconds --- this version is even much worse than the serial one


    ########################################
    # Measure the time for the elbow graph #
    ########################################

    ## Call one the function elbow graph once and check the time consumption
    start_time = time.time()
    elbow_results = elbow_graph_th(scaled_data, total_k, seed_value)
    print('--- %s seconds ---' % (time.time() - start_time))
    ## --- 338.7781488895416 seconds --- there is not a big difference compared with the serial version
    ## which took 346.7162780761719 seconds

    # 3. - Plot the first 2 dimensions of the clusters

    scatter = plt.scatter(scaled_data[:,0], scaled_data[:,1], cmap=plt.get_cmap('viridis'), c=res[:,p], label=res[:,p])
    plt.xlabel('price')
    plt.ylabel('speed')
    plt.title('Scatter Plot of two first dimensions with 2 clusters')
    plt.legend(*scatter.legend_elements(), loc='upper left', title='Clusters')
    plt.show()

    # 4. - Find the cluster with the highest average price and print it.

    res_df = pd.DataFrame(res, columns=np.append(df_without_cat.columns, 'cluster'))

    res_df_group = res_df.groupby('cluster')
    cluster_id = int(res_df_group.mean('price')['price'].idxmax())

    print('The cluster with the highest average price is {}'.format(cluster_id))

    # 5. - Print a heat map using the values of the clusters centroids.

    blues_pal = cm.get_cmap('Blues', 20)
    aggregations = {
        'price': 'mean',
        'speed': 'mean',
        'hd': 'mean',
        'ram': 'mean',
        'cores': 'mean',
        'screen': 'mean'
    }
    heatmap = res_df_group.agg(aggregations)
    g = sns.clustermap(np.transpose(heatmap), cmap=blues_pal)
    plt.show()