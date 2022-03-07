
library(dplyr)

# parameter k
# 1st step

X <- iris %>% dplyr::select(-Species) %>% as.matrix()
k <- 3

n <- nrow(X)
p <- ncol(X)

assig_cluster <- matrix(0, nrow=n, ncol=p+1)
centroids_not_equal <- TRUE
ite <- 1

centroids_index <- sample(x=n, size = k)
centroids <- X[centroids_index,]

while(centroids_not_equal) {
  print(centroids)
  print(sprintf("The iteration is: %s", ite))
  
  distance_cluster <- matrix(0, nrow=n, ncol=k)
  for (i in seq_len(k)){
    distance_cluster[, i] <- sqrt(rowSums((X[,]-t(replicate(n, centroids[i,])))^2))
  }
  
  cluster <- numeric(n)
  for (i in seq_len(n)){
    min_value_cluster <- which(distance_cluster[i,] == min(distance_cluster[i,]))
    cluster[i] <- sample(x=min_value_cluster, size=1)
  }
  assig_cluster <- cbind(X, cluster)
  
  new_centroids_index <- c()
  new_centroids <- matrix(0, nrow=k, ncol=p)
  for (i in seq_len(k)){
    elements_cluster <- X[which(assig_cluster[, p+1]==i),]
    centroidekth <- apply(X[which(assig_cluster[,p+1]==i),], MARGIN=2, FUN=mean)
    distance_centroide <- sqrt(rowSums((elements_cluster-t(replicate(nrow(elements_cluster), centroidekth)))^2))
    centroide_index <- which(distance_centroide== min(distance_centroide))
    new_centroids_index[i] <- sample(x=centroide_index, size=1)
    new_centroids[i,] <- X[centroide_index,]
  }
  if(isTRUE(all.equal(new_centroids_index, centroids_index))) {
    centroids_not_equal = FALSE
  } else {
    centroids = new_centroids
    centroids_index = new_centroids_index
  }
  ite = ite + 1
}

