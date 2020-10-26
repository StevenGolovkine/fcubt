# Perform Funclust on the data

# Load packages
library(fda)
library(Funclustering)
library(mclust)
library(tidyverse)

# Load data
argvals <- seq(0, 1, length.out = 101)
values_A <- t(as.matrix(read.csv('./data/scenario_2_A_smooth.csv', header = FALSE)))
values_B <- t(as.matrix(read.csv('./data/scenario_2_B_smooth.csv', header = FALSE)))
labels <- unname(as_vector(read.csv('./data/labels.csv', header = FALSE)))

# Perform Funclust
basis <- create.bspline.basis(rangeval = c(min(argvals), max(argvals)), nbasis = 25, norder = 3)
data_fd_A <- smooth.basis(argvals = argvals, y = values_A, fdParobj = basis)$fd
data_fd_B <- smooth.basis(argvals = argvals, y = values_B, fdParobj = basis)$fd

data_fd = list(data_fd_A, data_fd_B)
res <- tibble(n_cluster = numeric(), ARI = numeric())
n_clust <- seq(2, 8, 1)
for(i in n_clust){
  res_clust <- funclust(data_fd, K = i, thd = 0.2, epsilon = 1e-3,
                        nbInit = 2, nbIterInit = 5, nbIteration = 20)
  pred_labels <- res_clust$cls
  ARI <- adjustedRandIndex(labels, pred_labels)
  res <- add_row(res, n_cluster = length(unique(pred_labels)), ARI = ARI)
}
