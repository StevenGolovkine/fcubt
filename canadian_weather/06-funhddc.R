# Perform FunHDDC on the data

# Load packages
library(fda)
library(funHDDC)
library(tidyverse)

# Load data
argvals_A <- seq(0, 1, length.out = 365)
values_A <- as.matrix(read.csv('./data/canadian_temperature_daily_reduced.csv', header = FALSE))

argvals_B <- seq(0, 1, length.out = 364)
values_B <- as.matrix(read.csv('./data/canadian_precipitation_daily_reduced.csv', header = FALSE))

# FunHDDC
basis <- create.fourier.basis(c(0, 1), nbasis=65)
data_fd_A_smooth <- smooth.basis(argvals = argvals_A, y = t(values_A), fdParobj = basis)$fd
data_fd_B_smooth <- smooth.basis(argvals = argvals_B, y = t(values_B), fdParobj = basis)$fd

data_fd = list(data_fd_A_smooth, data_fd_B_smooth)
res <- funHDDC(data_fd, K = 2:10, model = c('AkjBkQkDk', 'AkjBQkDk', 'AkBkQkDk', 
                                            'ABkQkDk', 'AkBQkDk', 'ABQkDk'), init = 'random')
pred_labels <- res$class

# Save results
saveRDS(pred_labels, file = './results/results_weather_funhddc.rds')
