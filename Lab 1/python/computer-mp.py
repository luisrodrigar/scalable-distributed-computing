from itertools import repeat
from matplotlib import cm
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import utils

serial = __import__('computer-serial')

## Part two – Parallel implementation, multiprocessing

## 1.- Write a parallel version of you program using multiprocessing

def calc_new_centroids(X, assign_cluster, i, p):
    x_index_kth_cluster = np.where(assign_cluster[:,p] == i)[0]
    x_kth_cluster = X[x_index_kth_cluster,]
    return np.mean(x_kth_cluster, axis=0)


def calc_distance_centroids(X, centroids, i):  
    return np.sqrt(np.sum(np.square(np.subtract(X, np.atleast_2d(centroids[i,:]))), axis=1))


def calc_cluster_associations(distance_cluster, i):  
    return np.where(distance_cluster[i,] == np.min(distance_cluster[i, ]))[0][0]


def custom_kmeans_mp(X, k, seed_value):
    num_cores = int(mp.cpu_count()-1)
    pool = mp.Pool(num_cores)

    (n, p) = X.shape
    assign_cluster = np.zeros((n, p+1))
    centroids_not_equal = True
    ite = 0
    np.random.seed(seed_value)
    centroids_index = np.random.choice(range(n), k)
    centroids = X[centroids_index, :]
    try:
        while(centroids_not_equal):
            distance_cluster = pool.starmap(calc_distance_centroids, zip(repeat(X), repeat(centroids), range(k)))
            distance_cluster = np.array(distance_cluster).T
            cluster = pool.starmap(calc_cluster_associations, zip(repeat(distance_cluster), range(n)))
            assign_cluster = np.append(X, np.atleast_2d(cluster).T, axis=1)
            new_centroids = np.array(pool.starmap(calc_new_centroids, zip(repeat(X), repeat(assign_cluster), range(k), repeat(p))))
                
            if (new_centroids==centroids).all():
                centroids_not_equal = False
            else:
                centroids = new_centroids
            ite += 1
    except Exception as err:
        print(err)
    finally:
        pool.close()
    return assign_cluster


def calc_sum_sq_distances(kmeans_res, i):
    index_k_centroids = i-1
    res_X = kmeans_res[index_k_centroids]
    (_, p) = res_X.shape
    sum_sq_dist = 0
    for j in range(i):
        elements_cluster = res_X[np.where(res_X[:,p-1]==j),]
        centroidekth = np.mean(res_X[np.where(res_X[:,p-1]==j),], axis=1)
        distance_centroid = np.sum(np.square(elements_cluster-centroidekth), axis=0)
        sum_sq_dist += np.sum(distance_centroid)
    return sum_sq_dist


def elbow_graph_mp(data, total_k, seed_value):
    num_cores = int(mp.cpu_count()-1)
    pool = mp.Pool(num_cores)

    sum_sq_dist_total = np.zeros(total_k)
    k_iter = range(1, total_k+1)
    try:
        kmeans_res = pool.starmap(serial.custom_kmeans, zip(repeat(data), k_iter, repeat(seed_value)))
        sum_sq_dist_total = pool.starmap(calc_sum_sq_distances, zip(repeat(kmeans_res), k_iter))
    except Exception as err:
        print(err)
    finally:
        pool.close()
    return sum_sq_dist_total


if __name__== "__main__":
    df = utils.perform_dataset()
    df_without_cat = utils.tiny_data(df)
    scaled_data = utils.scale_data(df_without_cat)

    (n, p) = scaled_data.shape
    total_k = 10
    optimal_k = 2
    seed_value = 1234

    # 2. - Measure the time and optimize the program to get the fastest version you can.

    ####################################
    # Measure the time for the k-means #
    ####################################

    ## Call one the function custom_kmeans once and check the time consumption
    start_time = time.time()
    res = custom_kmeans_mp(scaled_data, optimal_k, seed_value)
    print('--- %s seconds ---' % (time.time() - start_time))
    ## --- 6.640674114227295 seconds --- it has improved the time consumption, 
    ## compared with the serial version which took 8.744184732437134 seconds

    ########################################
    # Measure the time for the elbow graph #
    ########################################

    ## Call one the function custom_kmeans once and check the time consumption
    start_time = time.time()
    elbow_results = elbow_graph_mp(scaled_data, total_k, seed_value)
    print('--- %s seconds ---' % (time.time() - start_time))
    ## --- 108.24986410140991 seconds --- it has improved considerably, compared with the serial version
    ## which took 346.7162780761719 seconds

    plt.plot(np.array(range(total_k))+1, elbow_results, '-ob')
    plt.xlabel('Number of clusters')
    plt.ylabel('Total Sum of Squares')
    plt.title('Elbow graph')
    plt.show()

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

    viridis_pal = cm.get_cmap('plasma_r', 20)
    aggregations = {
        'price': 'mean',
        'speed': 'mean',
        'hd': 'mean',
        'ram': 'mean',
        'cores': 'mean',
        'screen': 'mean'
    }
    heatmap = res_df_group.agg(aggregations)
    g = sns.clustermap(np.transpose(heatmap), cmap=viridis_pal)
    plt.show()


