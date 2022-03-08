library(dplyr)

X <- iris %>% dplyr::select(-Species) %>% as.matrix()
k <- 3

X_pca <- prcomp(X)

custom_kmeans <- function(X, k) {
  n <- nrow(X)
  p <- ncol(X)
  
  assig_cluster <- matrix(0, nrow=n, ncol=p+1)
  centroids_not_equal <- TRUE
  ite <- 1
  
  centroids_index <- sample(x=n, size = k)
  centroids <- X[centroids_index,]
  
  while(centroids_not_equal) {
    print(sprintf("The iteration is: %s", ite))
    
    distance_cluster <- matrix(0, nrow=n, ncol=k)
    for (i in seq_len(k)){
      distance_cluster[, i] <- sqrt(rowSums((X[,]-t(replicate(n, centroids[i,])))^2))
    }
    
    cluster <- numeric(n)
    for (i in seq_len(n)){
      min_value_cluster <- which(distance_cluster[i,] == min(distance_cluster[i,]))
      cluster[i] <- sample(x=list(min_value_cluster), size=1)[[1]]
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
    
    print(assig_cluster)
    plot(x=X_pca$x[,1], y=X_pca$x[,3], col=assig_cluster[,5], main="Cluster")
    
    if(isTRUE(all.equal(new_centroids, centroids))) {
      centroids_not_equal = FALSE
    } else {
      centroids = new_centroids
    }
    ite = ite + 1
  }
  return(assig_cluster)
}

res <- custom_kmeans(X, k)

