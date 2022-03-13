


# Part three – Parallel implementation, threading

# 1.- Write a parallel version of you program using multiprocessing

custom_kmeans_threads <- function(data, k, seed_value) {
  num_cores <- detectCores()-1
  par_cluster <- parallel::makeCluster(num_cores, type = "FORK")
  doParallel::registerDoParallel(par_cluster)
  
  n <- nrow(data)
  p <- ncol(data)
  assig_cluster <- matrix(0, nrow=n, ncol=p+1)
  centroids_not_equal <- TRUE
  ite <- 1
  set.seed(seed = seed_value)
  centroids_index <- sample(x=n, size = k)
  centroids <- rbind(data[centroids_index,])
  while(centroids_not_equal) {
    
    distance_cluster <- foreach(i = seq_len(k), .combine = 'cbind') %dopar% {
      sqrt(rowSums(sweep(data, MARGIN=2, STATS=as.array(centroids[i,]), FUN = "-")^2))
    }
    
    distance_min = parApply(cl=par_cluster, X=cbind(distance_cluster), MARGIN=1, FUN=min)
    sub_distance_min = -1 * (distance_cluster - distance_min)
    cluster = max.col(sub_distance_min)
    assig_cluster <- cbind(data, cluster)
    
    new_centroids <- foreach(i = seq_len(k), .combine = 'rbind') %dopar% {
      apply(rbind(data[which(assig_cluster[, p+1]==i),]), MARGIN=2, FUN=mean)
    }
    
    if(isTRUE(all.equal(new_centroids, centroids)) || k == 1) {
      centroids_not_equal = FALSE
    } else {
      centroids = rbind(new_centroids)
    }
    
    ite = ite + 1
  }
  autoStopCluster(par_cluster)
  return(assig_cluster)
}


