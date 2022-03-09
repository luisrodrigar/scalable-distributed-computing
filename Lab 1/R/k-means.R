library(dplyr)
library(rgl)

data <- read.csv("../python/computers_dev.csv")

X <- data %>% mutate(
  cd = ifelse(cd == "no", 0, 1),
  laptop = ifelse(laptop == "no", 0, 1)
) %>% dplyr::select(-id) %>% as.matrix()

n <- nrow(X)
p <- ncol(X)

# Scale computers data
scale_X <- scale(X)

# Part one – Serial version

# 1. Construct the elbow graph and find the optimal clusters number(k)
## OPTION A

# 2.- Implement the k-means algorithm

# This function returns the same index when the parameters is an scalar
# and returns a random index selected from the list passing as parameter
random_index <- function(indexes, seed_value) {
  if(length(indexes) == 1) {
    return(indexes)
  } else {
    set.seed(seed = seed_value)
    return(sample(x=indexes, size=1))
  }
}

custom_kmeans <- function(X, k, seed_value) {
  n <- nrow(X)
  p <- ncol(X)
  
  assig_cluster <- matrix(0, nrow=n, ncol=p+1)
  centroids_not_equal <- TRUE
  ite <- 1
  
  set.seed(seed = seed_value)
  centroids_index <- sample(x=n, size = k)
  centroids <- rbind(X[centroids_index,])
  
  while(centroids_not_equal) {
    distance_cluster <- matrix(0, nrow=n, ncol=k)
    for (i in seq_len(k)){
      distance_cluster[, i] <- sqrt(rowSums((X[,]-t(replicate(n, centroids[i,])))^2))
    }
    
    cluster <- numeric(n)
    for (i in seq_len(n)){
      min_value_cluster_indexes <- which(distance_cluster[i,] == min(distance_cluster[i,]))
      cluster[i] <- random_index(min_value_cluster_indexes, seed_value)
    }
    assig_cluster <- cbind(X, cluster)
    
    new_centroids_index <- c()
    new_centroids <- matrix(0, nrow=k, ncol=p)
    for (i in seq_len(k)){
      x_index_kth_cluster <- which(assig_cluster[, p+1]==i)
      x_kth_cluster <- rbind(X[x_index_kth_cluster,])
      kthcentroid <- apply(x_kth_cluster, MARGIN=2, FUN=mean)

      new_centroids[i,] <- kthcentroid
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
  for (i in seq_len(total_k)) {
    res_X <- custom_kmeans(X, i, seed_value)
    sum_sq_distance <- 0
    for(j in seq_len(i)) {
      elements_cluster <- rbind(res_X[which(res_X[, p+1]==j),])
      centroidekth <- apply(res_X[which(res_X[,p+1]==j),], MARGIN=2, FUN=mean)
      distance_centroid <- rowSums((elements_cluster-t(replicate(nrow(elements_cluster), centroidekth)))^2)
      sum_sq_distance <- sum_sq_distance + sum(distance_centroid)
    }
    sum_sq_dist_total[i] = sum_sq_distance
  }
  plot(x=seq_len(total_k), y=sum_sq_dist_total, type="l", col="blue", 
       xlab="Number of clusters", ylab="Total Sum of Squares")
  points(x=seq_len(total_k), y=sum_sq_dist_total)
  return(sum_sq_dist_total)
}

seed_value = 123456
elbow_graph(scale_X, seed_value = seed_value)

# 3.- Cluster the data using the optimum value using k-means.
optimal_k <- 2
res <- custom_kmeans(scale_X, optimal_k, seed_value)

# 4.- Measure time

start_time <- Sys.time()
res <- custom_kmeans(scale_X, optimal_k, seed_value)
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
elbow_graph(scale_X, seed_value = seed_value)
end_time <- Sys.time()
end_time - start_time

microbenchmark::microbenchmark(custom_kmeans(scale_X, 2, 12345), times=10)

# 5.- Plot the results of the elbow graph.

elbow_graph(scale_X, seed_value = seed_value)

# 6.- Plot the first 2 dimensions of the clusters

plot(x=scale_X[,1], y=scale_X[,2], col=res[,p+1], xlab="price", ylab="speed",
     main="Cluster with optimal k = 2")
legend_names <- paste("cluster", seq_len(optimal_k), sep=" ")
legend("topleft", col=seq_len(optimal_k), legend=legend_names, lwd=2, bty = "n",
       cex=0.75)

## Plot the three first principal components

colors_2 <- viridis::viridis(2)
cluster_colors <- ifelse(res[, p +1] == 1, colors_2[1], colors_2[2])
X_pca <- prcomp(X)
plot3d(X_pca$x[,1], X_pca$x[,2], X_pca$x[,3], pch = 30, col=cluster_colors)
legend3d("topright", legend = legend_names, col = colors_2, pch=19)


# 7.- Find the cluster with the highest average price and print it.

res_avg_cluster <- data.frame(res) %>% 
  group_by(cluster) %>% 
  summarise(mean_price = mean(price))

cluster_high_avg_price <- res_avg_cluster %>% 
  dplyr::filter(mean_price == max(mean_price)) %>% 
  dplyr::select(cluster)

print(sprintf("The cluster with the highest average price is %d", 
              cluster_high_avg_price[[1]]))

# 6.- Print a heat map using the values of the clusters centroids.

heatmap(x = X, scale = "none", col = res[,p+1], cexRow = 0.7, labRow=data$id)

# Part two – Parallel implementation, multiprocessing

