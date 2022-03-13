source('utils.R')

scale_X <- tidy_dataset(dev_dataset())
x <- nrow(scale_X)
p <- ncol(scale_X)

# Part one – Serial version

# 1. Construct the elbow graph and find the optimal clusters number(k)

# 2.- Implement the k-means algorithm

custom_kmeans <- function(data, k, seed_value) {
  n <- nrow(data)
  p <- ncol(data)
  assig_cluster <- matrix(0, nrow=n, ncol=p+1)
  centroids_not_equal <- TRUE
  ite <- 1
  set.seed(seed = seed_value)
  centroids_index <- sample(x=n, size = k)
  centroids <- rbind(data[centroids_index,])
  while(centroids_not_equal) {
    distance_cluster <- matrix(0, nrow=n, ncol=k)
    for (i in seq_len(k)){
      substracting_dist <- sweep(data, MARGIN=2, STATS=as.array(centroids[i,]), FUN = "-")
      distance_cluster[, i] <- sqrt(rowSums(substracting_dist^2))
    }

    cluster <- numeric(n)
    for (i in seq_len(n)){
      min_value_cluster_indexes <- which(distance_cluster[i,] == min(distance_cluster[i,]))
      if(length(min_value_cluster_indexes) > 1) {
        set.seed(seed = seed_value)
        min_value_cluster_indexes = sample(x=min_value_cluster_indexes, size=1)
      }
      cluster[i] = min_value_cluster_indexes
    }
    assig_cluster <- cbind(data, cluster)
    
    new_centroids <- matrix(0, nrow=k, ncol=p)
    for (i in seq_len(k)){
      new_centroids[i,] <- colMeans(rbind(data[which(assig_cluster[, p+1]==i),]))
    }

    if(isTRUE(all.equal(new_centroids, centroids))) {
      centroids_not_equal = FALSE
    } else {
      centroids = new_centroids
    }
    ite = ite + 1
  }
  return(assig_cluster)
}

elbow_graph <- function(X, total_k = 10, seed_value) {
  n <- nrow(X)
  p <- ncol(X)
  sum_sq_dist_total <- numeric(total_k)
  
  res_X <- lapply(seq_len(total_k), 
                  FUN=custom_kmeans, 
                  data=X, 
                  seed_value=seed_value)
  
  for (i in seq_len(total_k)) {
    sum_sq_distance <- 0
    for(j in seq_len(i)) {
      res_X_i = res_X[[i]]
      elements_cluster <- rbind(res_X_i[which(res_X_i[, p+1]==j),])
      centroidekth <- colMeans(elements_cluster)
      distance_centroid <- rowSums((sweep(elements_cluster, STATS=as.array(centroidekth), MARGIN=2, FUN = "-"))^2)
      sum_sq_distance <- sum_sq_distance + sum(distance_centroid)
    }
    sum_sq_dist_total[i] = sum_sq_distance
  }
  return(sum_sq_dist_total)
}

seed_value = 1234
total_k = 10

# 3.- Cluster the data using the optimum value using k-means.
optimal_k <- 2
res <- custom_kmeans(scale_X, optimal_k, seed_value)

# 4.- Measure time

####################################
# Measure the time for the k-means #
####################################

## Call the function k-means once and check the time consumption
start_time <- Sys.time()
custom_kmeans(scale_X, 2, 1234)
end_time <- Sys.time()
end_time - start_time
## Time difference of 5.407493 secs for 500,000 rows in dataset

## Call the function k-means ten times and check the time consumption
start_time <- Sys.time()
lapply(seq_len(10), 
       FUN=custom_kmeans, 
       data=scale_X, 
       seed_value=seed_value)
end_time <- Sys.time()
end_time - start_time
## Time Time difference of 3.550611 mins for 500,000 rows in dataset

########################################
# Measure the time for the elbow graph #
########################################

## Call the function elbow graph and check the time consumption
start_time <- Sys.time()
elbow_results <- elbow_graph(scale_X, seed_value = seed_value)
end_time <- Sys.time()
end_time - start_time
## Time difference of 3.131855 mins for 500,000 rows in dataset
## This makes sense as the lapply check above it took 3.55 min only for 
## assessing the k-means for each k

microbenchmark::microbenchmark(custom_kmeans(scale_X, 2, 12345), times=10, unit = "ns")

# 5.- Plot the results of the elbow graph.

plot(x=seq_len(total_k), y=elbow_results, type="l", col="blue", 
     xlab="Number of clusters", ylab="Total Sum of Squares")
points(x=seq_len(total_k), y=elbow_results)

# 6.- Plot the first 2 dimensions of the clusters

plot(x=scale_X[,1], y=scale_X[,2], col=res[,p+1], xlab="price", ylab="speed",
     main="Cluster with optimal k = 2")
legend_names <- paste("cluster", seq_len(optimal_k), sep=" ")
legend("topleft", col=seq_len(optimal_k), legend=legend_names, lwd=2, bty = "n",
       cex=0.75)

## Plot the three first principal components

colors_2 <- viridis::viridis(2)
cluster_colors <- ifelse(res[, p +1] == 1, colors_2[1], colors_2[2])
X_pca <- prcomp(scale_X)
plot3d(X_pca$x[,1], X_pca$x[,2], X_pca$x[,3], pch = 30, col=cluster_colors)
legend3d("topright", legend = legend_names, col = colors_2, pch=19)

# 7.- Find the cluster with the highest average price and print it

res_group_cluster <- data.frame(res) %>% 
  group_by(cluster)

cluster_high_avg_price <- res_group_cluster %>% 
  summarise(mean_price = mean(price)) %>% 
  dplyr::filter(mean_price == max(mean_price)) %>% 
  dplyr::select(cluster)

print(sprintf("The cluster with the highest average price is %d", 
              cluster_high_avg_price[[1]]))

# 6.- Print a heat map using the values of the clusters centroids

cluster_summary <- res_group_cluster %>% 
  summarise(
    price = mean(price),
    speed = mean(speed),
    hd = mean(hd),
    ram = mean(ram),
    cores = mean(cores),
    screen = mean(screen)
  ) %>% dplyr::select(-cluster) %>% as.data.frame

heatmap.2(x = t(cluster_summary), scale = "none", trace="none", cexRow = 0.7,
        col= brewer.pal(8, "Oranges"), density.info = "none")



