# Perform FunHDDC on the data

# Load packages
library(fda)
library(funHDDC)
library(mclust)
library(tidyverse)

# Load data
argvals <- seq(0, 1, length.out = 100)
values <- t(as.matrix(read.csv('./data/scenario_1.csv', header = FALSE)))
labels <- unname(as_vector(read.csv('./data/labels.csv', header = FALSE)))

# FunHDDC
basis <- create.bspline.basis(rangeval = c(min(argvals), max(argvals)), nbasis = 25, norder = 3)
data_fd <- smooth.basis(argvals = argvals, y = values, fdParobj = basis)$fd
res_clust <- funHDDC(data_fd, K = 2:10, model = 'ABkQkDk', threshold = 0.2,
                     itermax = 200, eps = 1e-3, init = 'kmeans', criterion = 'bic')
pred_labels <- res_clust$class
ARI <- adjustedRandIndex(labels, pred_labels)
