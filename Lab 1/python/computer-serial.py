from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import utils

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

def elbow_graph(X, total_k, seed_value):
    (_, p) = X.shape
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
    return sum_sq_dist_total


if __name__ == "__main__":
    df_without_cat = utils.tiny_data(utils.perform_dataset())
    scaled_data = utils.scale_data(df_without_cat)
    (n, p) = scaled_data.shape
    optiomal_k = 2
    total_k = 10
    seed_value = 1234
    
    ## 4. - Measure time

    ####################################
    # Measure the time for the k-means #
    ####################################

    ## Call one the function custom_kmeans once and check the time consumption
    start_time = time.time()
    res = custom_kmeans(scaled_data, optiomal_k, seed_value)
    print('--- %s seconds (k-means) ---' % (time.time() - start_time))
    ## Time spent: --- 8.744184732437134 seconds (k-means) --- for 500,000 rows in the dataset

    ## Call the function custom_kmeans ten times for each k and check the time consumption
    start_time = time.time()
    kmeans_list = [custom_kmeans(scaled_data, k, seed_value) for k in range(1, total_k+1)]
    print('--- %s seconds (k-means iter.) ---' % (time.time() - start_time))
    ## --- 345.5383791923523 seconds (k-means iter.) ---for 500,000 rows in the dataset

    ########################################
    # Measure the time for the elbow graph #
    ########################################

    ## Call the function elbow graph once and check the time consumption
    start_time = time.time()
    elbow_results = elbow_graph(scaled_data, total_k, seed_value)
    print('--- %s seconds (elbow graph) ---' % (time.time() - start_time))
    ## --- 346.7162780761719 seconds (elbow graph) --- for 500,000 rows in the dataset

    ## 5. - Plot the results of the elbow graph.

    plt.plot(np.array(range(total_k))+1, elbow_results, '-ob')
    plt.xlabel('Number of clusters')
    plt.ylabel('Total Sum of Squares')
    plt.title('Elbow graph')
    plt.show()

    ## 6. - Plot the first 2 dimensions of the clusters
    scatter = plt.scatter(scaled_data[:,0], scaled_data[:,1], cmap=plt.get_cmap('viridis'), c=res[:,p], label=res[:,p])
    plt.xlabel('price')
    plt.ylabel('speed')
    plt.title('Scatter Plot of two first dimensions with 2 clusters')
    plt.legend(*scatter.legend_elements(), loc='upper left', title='Clusters')
    plt.show()

    ## 7. - Find the cluster with the highest average price and print it.

    res_df = pd.DataFrame(res, columns=np.append(df_without_cat.columns, 'cluster'))

    res_df_group = res_df.groupby('cluster')
    cluster_id = int(res_df_group.mean('price')['price'].idxmax())

    print('The cluster with the highest average price is {}'.format(cluster_id))

    ## 8. - Print a heat map using the values of the clusters centroids.

    viridis_pal = cm.get_cmap('viridis_r', 20)
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


