library(doParallel)
library(parallel)

# Part two – Parallel implementation, multiprocessing
# 1.- Write a parallel version of you program using multiprocessing


custom_kmeans_parallel <- function(X, k, seed_value) {
  numCores <- detectCores()
  registerDoParallel(numCores) 
  
  n <- nrow(X)
  p <- ncol(X)
  assig_cluster <- matrix(0, nrow=n, ncol=p+1)
  centroids_not_equal <- TRUE
  ite <- 1
  set.seed(seed = seed_value)
  centroids_index <- sample(x=n, size = k)
  centroids <- rbind(X[centroids_index,])
  while(centroids_not_equal) {
    distance_cluster <- foreach(i = seq_len(k), .combine = 'cbind')  %dopar%  {
      sqrt(rowSums((X[,]-t(replicate(n, centroids[i,])))^2))
    }
    
    cluster <- foreach(i = seq_len(n), .combine = 'c')  %dopar%  {
      min_value_cluster_indexes <- which(
        distance_cluster[i,] == min(distance_cluster[i,]))
      random_index(min_value_cluster_indexes, seed_value)
    }
    
    assig_cluster <- cbind(X, cluster)
    new_centroids <- matrix(0, nrow=k, ncol=p)
    for (i in seq_len(k)){
      x_index_kth_cluster <- which(assig_cluster[, p+1]==i)foreach
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

start_time_multi <- Sys.time()
knn_multi <- parLapply(cl = clust, X=scale_X, k=optimal_k, seed_value=12345, fun = custom_kmeans)
end_time_multi <- Sys.time()