# Perform FunHDDC on the data

# Load packages
library(fda)
library(funHDDC)
library(mclust)
library(tidyverse)

# Load data
argvals <- seq(0, 1, length.out = 101)
values_A <- t(as.matrix(read.csv('./data/scenario_3_A_smooth.csv', header = FALSE)))
values_B <- t(as.matrix(read.csv('./data/scenario_3_B_smooth.csv', header = FALSE)))
labels <- unname(as_vector(read.csv('./data/labels.csv', header = FALSE)))

# FunHDDC
basis <- create.bspline.basis(rangeval = c(min(argvals), max(argvals)), nbasis = 25, norder = 3)
data_fd_A <- smooth.basis(argvals = argvals, y = values_A, fdParobj = basis)$fd
data_fd_B <- smooth.basis(argvals = argvals, y = values_B, fdParobj = basis)$fd

data_fd = list(data_fd_A, data_fd_B)
res_clust <- funHDDC(data_fd, K = 1:10, model = 'ABkQkDk', threshold = 0.2,
                     itermax = 200, eps = 1e-3, init = 'kmeans', criterion = 'bic')

pred_labels <- res_clust$class
ARI <- adjustedRandIndex(labels, pred_labels)


res_clust <- funHDDC(data_fd, K = 1:10, model = 'AkjBkQkDk', threshold = 0.2,
                     itermax = 200, eps = 1e-3, init = 'kmeans', criterion = 'bic')

pred_labels <- res_clust$class
ARI <- adjustedRandIndex(labels, pred_labels)



