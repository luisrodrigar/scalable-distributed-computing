library(dplyr)
library(rgl)
library(RColorBrewer)
library(gplots)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)

tidy_dataset <- function(data) {
  # Delete the categorical variables as it is not possible to apply k-means on them
  X <- data %>% mutate(
    cd = ifelse(cd == "no", 0, 1),
    laptop = ifelse(laptop == "no", 0, 1)
  ) %>% dplyr::select(-id)
  
  X_without_cat <- X %>% dplyr::select(-cd, -laptop, -trend) %>% as.matrix()
  
  n <- nrow(X_without_cat)
  p <- ncol(X_without_cat)
  
  # Scale computers data
  scale_X <- scale(X_without_cat)
  return(scale_X)
}

dev_dataset <- function() {
  return(read.csv("../python/computers_dev.csv"))
}

performance_dataset <- function() {
  return(read.csv("../python/computers_perform.csv"))
}

heavy_dataset <- function() {
  return(read.csv("../python/computers.csv"))
}

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