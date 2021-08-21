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
		simu = KarhunenLoeve('wiener', n_functions=n_features)
		simu.new(n_obs=N, n_clusters=n_clusters, 
				 centers=centers,  
				 cluster_std=cluster_std)

		mean_data = np.vstack(
			[mean_function_1(simu.data.argvals['input_dim_0']),
			 mean_function_2(simu.data.argvals['input_dim_0'])])

		new_values = np.vstack(
			[simu.data.values[:int(N/2), :] + mean_data[0, :],
			 simu.data.values[int(N/2):, :] + mean_data[0, :],
			 simu.data.values[:int(N/2), :] + mean_data[1, :],
			 simu.data.values[int(N/2):, :] + mean_data[1, :],
			 simu.data.values[int(N/2):, :] + mean_data[1, :] - 15 *
				np.linspace(0, 1, 100)
			])

		data = DenseFunctionalData(simu.data.argvals, new_values)
		labels = np.hstack([simu.labels, simu.labels + 2, 
							np.repeat(4, int(N/2))])
		
		# Save the reduced data
		with open(f'./data/scenario_1_{i}.pkl', 'wb') as f:
			pickle.dump(data, f)
		with open('./data/labels.pkl', 'wb') as f:
			pickle.dump(labels, f)
		
		# Save as CSV for R methods
		np.savetxt(f'./data/scenario_1_{i}.csv', data.values, delimiter=',')
		np.savetxt('./data/labels.csv', labels, delimiter=',')


if __name__ == "__main__":
	main()