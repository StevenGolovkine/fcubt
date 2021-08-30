#!/usr/bin/env python
# -*-coding:utf8 -*

# Load packages
import numpy as np
import pickle

from FDApy.representation.simulation import KarhunenLoeve
from FDApy.representation.functional_data import DenseFunctionalData

# Define parameters of the simulation
N = 300
n_features = 3
n_clusters = 2
centers = np.array([[0, 0], [0, 0], [0, 0]])
cluster_std = np.array([[4, 1], [2.66, 0.66], [1.33, 0.33]])

# Define mean functions
mean_function_1 = lambda x: 20 / (1 + np.exp(-x))
mean_function_2 = lambda x: -25 / (1 + np.exp(-x))

def main():
	for i in np.arange(500):
		print(f'Simulation {i}')
		A = np.array([np.random.laplace(0, 2 * np.sqrt(2), size = 150),
			  np.random.laplace(0, 4 * np.sqrt(2) / 3, size = 150),
			  np.random.laplace(0, 2 * np.sqrt(2) / 3, size = 150)
			 ]).T
		B = np.array([np.random.laplace(0, 1 / np.sqrt(2), size = 150),
			  np.random.laplace(0, np.sqrt(2) / 3, size = 150),
			  np.random.laplace(0, 1 / (np.sqrt(2) * 3), size = 150)
			 ]).T
		new_coefs = np.concatenate([A, B])

		# Simulation one scenario
		simu = KarhunenLoeve('wiener', n_functions=n_features)
		simu.new(n_obs=N, n_clusters=n_clusters, centers=centers, 
			cluster_std=cluster_std)

		values = np.matmul(new_coefs, simu.basis.values)

		mean_data = np.vstack(
			[mean_function_1(simu.data.argvals['input_dim_0']),
			 mean_function_2(simu.data.argvals['input_dim_0'])])

		new_values = np.vstack(
			[values[:int(N/2), :] + mean_data[0, :],
			 values[int(N/2):, :] + mean_data[0, :],
			 values[:int(N/2), :] + mean_data[1, :],
			 values[int(N/2):, :] + mean_data[1, :],
			 values[int(N/2):, :] + mean_data[1, :] - 15 * np.linspace(0, 1, 
				100)
			 ])

		data = DenseFunctionalData(simu.data.argvals, new_values)
		labels = np.hstack([simu.labels, simu.labels + 2, 
			np.repeat(4, int(N/2))])
		
		# Save the reduced data
		with open(f'./data/scenario_1_{i}_review.pkl', 'wb') as f:
			pickle.dump(data, f)
		with open('./data/labels_review.pkl', 'wb') as f:
			pickle.dump(labels, f)


if __name__ == "__main__":
	main()