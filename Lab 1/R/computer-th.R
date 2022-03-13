source('utils.R')
source("computer-serial.R")

scale_X <- tidy_dataset(performance_dataset())
x <- nrow(scale_X)
p <- ncol(scale_X)
optimal_k = 2
# Part three – Parallel implementation, threading

# 1.- Write a parallel version of you program using multiprocessing

custom_kmeans_th <- function(data, k, seed_value) {
  num_cores <- parallel::detectCores()
  par_cluster <- parallel::makeForkCluster(as.integer(num_cores-1))
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
    
    distance_cluster <- foreach(i = seq_len(k), .combine = 'cbind', .noexport = 'par_cluster') %dopar% {
      sqrt(rowSums(sweep(data, MARGIN=2, STATS=as.array(centroids[i,]), FUN = "-")^2))
    }
    
    distance_min = parApply(cl=par_cluster, X=cbind(distance_cluster), MARGIN=1, FUN=min)
    sub_distance_min = -1 * (distance_cluster - distance_min)
    cluster = max.col(sub_distance_min)
    assig_cluster <- cbind(data, cluster)
    
    new_centroids <- foreach(i = seq_len(k), .combine = 'rbind', .noexport = 'par_cluster') %dopar% {
      colMeans(rbind(data[which(assig_cluster[, p+1]==i),]))
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

elbow_graph_th <- function(X, total_k = 10, seed_value) {
  n <- nrow(X)
  p <- ncol(X)
  num_cores <- detectCores()-1
  
  kmeans_data <- mclapply(X=seq_len(total_k), 
                          FUN=custom_kmeans,
                          data=X, 
                          seed_value=1234,
                          mc.cores = num_cores)
  
  sum_sq_dist_total <- foreach(i = seq_len(total_k), .combine="c", 
                               .noexport='par_cluster') %:%
    foreach(j = seq_len(i), .combine="+") %dopar% {
      res_data <- kmeans_data[[i]]
      elements_cluster <- rbind(res_data[which(res_data[, p+1]==j),])
      centroidekth <- colMeans(elements_cluster)
      dista_matrix <- sweep(elements_cluster, STATS=as.array(centroidekth), MARGIN=2, FUN = "-")
      dist_centroid <- rowSums(dista_matrix^2)
      sum(dist_centroid)
    }
  on.exit(autoStopCluster(par_cluster))
  
  return(sum_sq_dist_total)
}

# 2. - Measure the time and optimize the program to get the fastest version you can.

####################################
# Measure the time for the k-means #
####################################

## Call the function k-means once and check the time consumption
start_time <- Sys.time()
kmeans_th <- custom_kmeans_th(scale_X, optimal_k, 1234)
end_time <- Sys.time()
end_time - start_time
## Time difference of 13.32509 secs for 500,000 rows in dataset
## There is no improvement as the serial version took 5.407493 secs
## It happens often that parallelization for quick tasks is not worthy

## Call the serial function k-means ten times (FORKING) and check the time consumption
num_cores <- detectCores()-1
start_time <- Sys.time()
kmeans_list <- mclapply(X=seq_len(total_k), 
                        FUN=custom_kmeans,
                        data=scale_X, 
                        seed_value=1234,
                        mc.cores = num_cores)
end_time <- Sys.time()
end_time - start_time
## Time difference of 1.541406 mins for 500,000 rows in dataset
## It is slightly better than the mp (SOKET) version
## This is something expected as this procedure copies the full data in the process

###############################################
# print("Measure the time for the elbow graph #
###############################################

## Call the function elbow graph once and check the time consumption
start_time <- Sys.time()
elbow_results <- elbow_graph_th(scale_X, 10, 1234)
end_time <- Sys.time()
end_time - start_time
## Time difference of 1.584283 mins for 500,000 rows in dataset
## Slightly greater than the time obtained in the previous test as the 
## mclapply is inside the elbow praph function using threads (fork)

# 3. - Plot the first 2 dimensions of the clusters

plot(x=scale_X[,1], y=scale_X[,2], col=kmeans_th[,p+1], xlab="price", ylab="speed",
     main="Cluster with optimal k = 2 (threading)")
legend_names <- paste("cluster", seq_len(optimal_k), sep=" ")
legend("topleft", col=seq_len(optimal_k), legend=legend_names, lwd=2, bty = "n",
       cex=0.75)

# 4. - Find the cluster with the highest average price and print it.

res_group_cluster <- data.frame(kmeans_th) %>% 
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
          col= brewer.pal(8, "Greens"), density.info = "none")
