# Perform Funclust on the data

# Load packages
library(fda)
library(Funclustering)
library(mclust)
library(tidyverse)

# Load data
argvals <- seq(0, 1, length.out = 100)
values <- t(as.matrix(read.csv('./data/scenario_1.csv', header = FALSE)))
labels <- unname(as_vector(read.csv('./data/labels.csv', header = FALSE)))

# Perform Funclust
basis <- create.bspline.basis(rangeval = c(min(argvals), max(argvals)), nbasis = 25, norder = 3)
data_fd <- smooth.basis(argvals = argvals, y = values, fdParobj = basis)$fd

res <- tibble(n_cluster = numeric(), ARI = numeric())
n_clust <- seq(2, 8, 1)
for(i in n_clust){
  res_clust <- funclust(data_fd, K = i, thd = 0.2, epsilon = 1e-3)
  pred_labels <- res_clust$cls
  ARI <- adjustedRandIndex(labels, pred_labels)
  res <- add_row(res, n_cluster = length(unique(pred_labels)), ARI = ARI)
}

