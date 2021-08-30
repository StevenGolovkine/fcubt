#!/usr/bin/env python
# -*-coding:utf8 -*

# Load packages
import numpy as np
import pickle

from FDApy.representation.simulation import KarhunenLoeve
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

from joblib import Parallel, delayed

import multiprocessing

num_cores = multiprocessing.cpu_count()

# Define parameters of the simulation for the curves
N = 500
n_features = 3
n_clusters = 5
centers = np.array([[3, 1, 1, -2, -2], 
					[2, -2, -2, 0, 0], 
					[1, 0, 0, -1, -1]])
cluster_std = np.array([[0.5, 0.5, 0.4, 1, 0.2], 
						[1.66, 1, 0.8, 2, 0.5], 
						[1.33, 1, 0.8, 2, 0.5]])
labels = np.repeat([0, 1, 2, 3, 4], int(N / n_clusters))

# Define parameters of the simulation for the images
N = 500
n_features = 2 # x2 because of the tensor product
n_clusters = 5
centers = np.array([[4, 4, -3, 0, 0], 
					[0, 0, -4, 2, 2], 
					[0, 0, 0, 0, 0], 
					[-2, -2, 0, 1, 1]])
cluster_std = np.array([[1, 0.8, 1, 0.1, 2], 
						[0.5, 0.7, 0.5, 0.1, 1],
						[0.1, 0.08, 0.1, 0.05, 0.2], 
						[0.05, 0.07, 0.05, 0.025, 0.1]])

def simulation(i):
	print(f'Simulation {i}')
	# Generate the curves
	simu_curves = KarhunenLoeve('wiener', n_functions=n_features)
	simu_curves.new(n_obs=N, n_clusters=n_clusters, 
		centers=centers, cluster_std=cluster_std)
	simu_curves.add_noise(var_noise=0.05)

	# Generate the images
	simu_images = KarhunenLoeve('wiener', n_functions=n_features, 
		dimension='2D')
	simu_images.new(n_obs=N, n_clusters=n_clusters, 
		centers=centers, cluster_std=cluster_std)
	simu_images.add_noise(var_noise=0.05)

	# Generate the data
	data_fd = MultivariateFunctionalData([simu_curves.noisy_data,
										  simu_images.noisy_data])

	# Save the data
	with open(f'./data/scenario_4_{i}.pkl', 'wb') as f:
		pickle.dump(data_fd, f)
	with open('./data/labels.pkl', 'wb') as f:
		pickle.dump(labels, f)

if __name__ == "__main__":
	results = Parallel(n_jobs=num_cores)(delayed(simulation)(i) 
		for i in np.arange(100))
