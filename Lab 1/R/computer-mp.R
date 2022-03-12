library(parallel)
library(foreach)
library(doParallel)
source("computer-serial.R")

# Part two – Parallel implementation, multiprocessing

# 1.- Write a parallel version of you program using multiprocessing

autoStopCluster <- function(cl) {
  stopifnot(inherits(cl, "cluster"))
  env <- new.env()
  env$cluster <- cl
  attr(cl, "gcMe") <- env
  reg.finalizer(env, function(e) {
    message(capture.output(print(e$cluster)))
    try(parallel::stopCluster(e$cluster), silent = FALSE)
  })
  cl
}

custom_kmeans_parallel <- function(k, data, seed_value) {
  num_cores <- detectCores()-1
  par_cluster <- parallel::makeCluster(num_cores)
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

elbow_graph_parallel <- function(X, total_k = 10, seed_value) {
  n <- nrow(X)
  p <- ncol(X)
  num_cores <- detectCores()-1
  par_cluster <- parallel::makeCluster(num_cores)
  doParallel::registerDoParallel(par_cluster)
  clusterEvalQ(par_cluster, {
    library(parallel)
    library(foreach)
    library(doParallel)
  })
  clusterExport(cl=par_cluster, "custom_kmeans_parallel", envir = environment())
  clusterExport(cl=par_cluster, "autoStopCluster", envir = environment())
  
  kmeans_data <- parLapply(cl=par_cluster, 
                           seq_len(total_k), 
                           custom_kmeans_parallel, 
                           data=X, 
                           seed_value=1234)
  
  sum_sq_dist_total <- foreach(i = seq_len(total_k), .combine="c") %:%
    foreach(j = seq_len(i), .combine="+") %dopar% {
      res_data <- kmeans_data[[i]]
      elements_cluster <- rbind(res_data[which(res_data[, p+1]==j),])
      centroidekth <- apply(res_data[which(res_data[,p+1]==j),], MARGIN=2, FUN=mean)
      dista_matrix <- sweep(elements_cluster, MARGIN=2, STATS=as.array(centroidekth), FUN = "-")
      dist_centroid <- rowSums(dista_matrix^2)
      sum(dist_centroid)
    }
  
  autoStopCluster(par_cluster)
  plot(x=seq_len(total_k), y=sum_sq_dist_total, type="l", col="blue", 
       xlab="Number of clusters", ylab="Total Sum of Squares")
  points(x=seq_len(total_k), y=sum_sq_dist_total)
  return(sum_sq_dist_total)
}

# 2. - Measure the time and optimize the program to get the fastest version you can.

print("Measure the time for the k-mean ")

print("Serial version:") 

start_time_multi <- Sys.time()
kmeans_opt <- custom_kmeans(2, scale_X, 1234)
end_time_multi <- Sys.time()
end_time_multi-start_time_multi

print("Parallel multiprocessing:") 

start_time_multi <- Sys.time()
kmeans_opt_parallel <- custom_kmeans_parallel(2, scale_X, 1234)
end_time_multi <- Sys.time()
end_time_multi-start_time_multi

print("Measure the time for the elbow graph")

print("Serial version:") 

start_time_multi <- Sys.time()
elbow_graph_multi <- elbow_graph(scale_X, 10, 1234)
end_time_multi <- Sys.time()
end_time_multi-start_time_multi

print("Parallel multiprocessing:") 

start_time_multi <- Sys.time()
elbow_graph_multi <- elbow_graph_parallel(scale_X, 10, 1234)
end_time_multi <- Sys.time()
end_time_multi-start_time_multi

# 3. - Plot the first 2 dimensions of the clusters

plot(x=scale_X[,1], y=scale_X[,2], col=kmeans_multi[,p+1], xlab="price", ylab="speed",
     main="Cluster with optimal k = 2")
legend_names <- paste("cluster", seq_len(optimal_k), sep=" ")
legend("topleft", col=seq_len(optimal_k), legend=legend_names, lwd=2, bty = "n",
       cex=0.75)

# 4. - Find the cluster with the highest average price and print it.

res_group_cluster <- data.frame(kmeans_opt_parallel) %>% 
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
