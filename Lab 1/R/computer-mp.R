source('utils.R')
source("computer-serial.R")

scale_X <- tidy_dataset(performance_dataset())

# Part two – Parallel implementation, multiprocessing

# 1.- Write a parallel version of you program using multiprocessing

custom_kmeans_mp <- function(data, k, seed_value) {
  num_cores <- parallel::detectCores()
  par_cluster <- parallel::makeCluster(as.integer(num_cores/2))
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

    new_centroids <- foreach(i = seq_len(k), .combine = 'rbind',
                             .noexport='par_cluster') %dopar% {
      apply(rbind(data[which(assig_cluster[, p+1]==i),]), MARGIN=2, FUN=mean)
    }
    
    if(isTRUE(all.equal(new_centroids, centroids)) || k == 1) {
      centroids_not_equal = FALSE
    } else {
      centroids = rbind(new_centroids)
    }
    
    ite = ite + 1
  }
  on.exit(autoStopCluster(par_cluster))
  return(assig_cluster)
}

elbow_graph_mp <- function(X, total_k = 10, seed_value) {
  n <- nrow(X)
  p <- ncol(X)
  num_cores <- detectCores()
  par_cluster <- parallel::makeCluster(as.integer(num_cores-1))
  doParallel::registerDoParallel(par_cluster)
  
  kmeans_data <- parLapply(cl=par_cluster, 
                           seq_len(total_k), 
                           custom_kmeans, 
                           data=X, 
                           seed_value=1234)
  
  sum_sq_dist_total <- foreach(i = seq_len(total_k), .combine="c", 
                               .noexport='par_cluster') %:%
    foreach(j = seq_len(i), .combine="+") %dopar% {
      res_data <- kmeans_data[[i]]
      elements_cluster <- rbind(res_data[which(res_data[, p+1]==j),])
      centroidekth <- apply(res_data[which(res_data[,p+1]==j),], MARGIN=2, FUN=mean)
      dista_matrix <- sweep(elements_cluster, MARGIN=2, STATS=as.array(centroidekth), FUN = "-")
      dist_centroid <- rowSums(dista_matrix^2)
      sum(dist_centroid)
    }
  on.exit(autoStopCluster(par_cluster))
  
  return(sum_sq_dist_total)
}

# 2. - Measure the time and optimize the program to get the fastest version you can.

print("###############################")
print("Measure the time for the k-mean")
print("###############################")

## Call the function k-means once and check the time consumption
start_time <- Sys.time()
kmeans_multi <- custom_kmeans_mp(scale_X, 2, 1234)
end_time <- Sys.time()
end_time - start_time
## Time difference of 12.35497 secs for 500,000 rows in dataset
## There is no improvement as the serial version took 5.312093 secs
## It happens often that parallelization for quick tasks is not worth

## Call the serial function k-means ten times and check the time consumption
num_cores <- parallel::detectCores()
par_cluster <- parallel::makeCluster(as.integer(num_cores/2))
doParallel::registerDoParallel(par_cluster)

start_time <- Sys.time()
parLapply(par_cluster, seq_len(10), 
          fun=custom_kmeans, 
          data=scale_X, 
          seed_value=seed_value)
end_time <- Sys.time()
end_time - start_time

autoStopCluster(par_cluster)
## Time difference of 1.416775 mins for 500,000 rows in dataset
## It is a great improvement as it took 3.550611 mins for the serial version of lapply

## Call the multiprocessing function k-means ten times and check the time consumption
par_cluster <- parallel::makeCluster(as.integer(num_cores/2))
doParallel::registerDoParallel(par_cluster)
clusterExport(par_cluster, 'autoStopCluster', envir = environment())
clusterEvalQ(par_cluster, {
  library('parallel')
  library('doParallel')
})
start_time <- Sys.time()
parLapply(par_cluster, seq_len(10), 
          fun=custom_kmeans_parallel, 
          data=scale_X, 
          seed_value=seed_value)
end_time <- Sys.time()
end_time - start_time

autoStopCluster(par_cluster)
## Time difference of 7.233679 mins for 500,000 rows in dataset
## This version is even much worse than the serial one
## This could happen because in parallel programming is not recommended
## to apply in two levels (parLapply is one level and the function itself the other)

print("####################################")
print("Measure the time for the elbow graph")
print("####################################")

print("Parallel multiprocessing:")

## Call the multiprocessing function elbow graph and check the time consumption
start_time_multi <- Sys.time()
elbow_graph_multi <- elbow_graph_parallel(scale_X, 10, 1234)
end_time_multi <- Sys.time()
end_time_multi-start_time_multi
## Time difference of 1.119786 mins for 500,000 rows in dataset
## It considerably reduces the time to process the data compared with the serial version
## which took 3.598459 mins to process the same data

# 3. - Plot the first 2 dimensions of the clusters

plot(x=scale_X[,1], y=scale_X[,2], col=kmeans_multi[,p+1], xlab="price", ylab="speed",
     main="Cluster with optimal k = 2")
legend_names <- paste("cluster", seq_len(optimal_k), sep=" ")
legend("topleft", col=seq_len(optimal_k), legend=legend_names, lwd=2, bty = "n",
       cex=0.75)

# 4. - Find the cluster with the highest average price and print it.

res_group_cluster <- data.frame(kmeans_multi) %>% 
  group_by(cluster)

cluster_high_avg_price <- res_group_cluster %>% 
  summarise(mean_price = mean(price)) %>% 
  dplyr::filter(mean_price == max(mean_price)) %>% 
  dplyr::select(cluster)

print(sprintf("The cluster with the highest average price is %d", 
              cluster_high_avg_price[[1]]))

# 5. - Print a heat map using the values of the clusters centroids.

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
          col= brewer.pal(8, "Blues"), density.info = "none")
