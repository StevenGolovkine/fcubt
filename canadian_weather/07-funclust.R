# Perform Funclust on the data

# Load packages
library(fda)
library(Funclustering)
library(tidyverse)

# Load data
argvals_A <- seq(0, 1, length.out = 365)
values_A <- as.matrix(read.csv('./data/canadian_temperature_daily_reduced.csv', header = FALSE))

argvals_B <- seq(0, 1, length.out = 364)
values_B <- as.matrix(read.csv('./data/canadian_precipitation_daily_reduced.csv', header = FALSE))

# Perform Funclust
basis <- create.fourier.basis(c(0, 1), nbasis = 65)
data_fd_A_smooth <- smooth.basis(argvals = argvals_A, y = t(values_A), fdParobj = basis)$fd
data_fd_B_smooth <- smooth.basis(argvals = argvals_B, y = t(values_B), fdParobj = basis)$fd

data_fd = list(data_fd_A_smooth, data_fd_B_smooth)
res_funclust <- list()
for(i in seq(1, 10, 1)){
  res_funclust[i] <- list(funclust(data_fd, K = i, increaseDimension = TRUE))
}

# Larger BIC
bic_idx <- which.max(res_funclust %>% map_dbl(~ .x$bic))
pred_labels <- res_funclust[[bic_idx]]$cls

# Save results
saveRDS(pred_labels, file = './results/results_weather_funclust.rds')
