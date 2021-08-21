#!/usr/bin/env python
# -*-coding:utf8 -*

# Load packages
import numpy as np
import pickle

from FDApy.representation.simulation import Brownian
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

from joblib import Parallel, delayed

import multiprocessing

num_cores = multiprocessing.cpu_count()

# Define parameters of the simulation
N1, N2, N3, N4, N5 = 200, 200, 200, 200, 200
M = 101
hurst1, hurst2 = 0.9, 0.8
x = np.linspace(1, 21, M)
labels = np.repeat([0, 1, 2, 3, 4], repeats=(N1, N2, N3, N4, N5))

# Define mean functions
def h(x, a):
	return 6 - np.abs(x - a)

h_1 = lambda x: h(x, 7) / 4 if h(x, 7) > 0 else 0
h_2 = lambda x: h(x, 15) / 4 if h(x, 15) > 0 else 0
h_3 = lambda x: h(x, 11) / 4 if h(x, 11) > 0 else 0

def simulation(i):
	print(f'Simulation {i}')
	A = np.zeros((N1 + N2 + N3 + N4 + N5, M))
	B = np.zeros((N1 + N2 + N3 + N4 + N5, M))
	for idx in range(N1 + N2 + N3 + N4 + N5):
		h1 = np.array([h_1(i) for i in x])
		h2 = np.array([h_2(i) for i in x])
		h3 = np.array([h_3(i) for i in x])
			
		brownian = Brownian(name='fractional')
		brownian.new(1, argvals=np.linspace(0, 2, 2 * M), hurst=hurst1)
		rand_part1 = brownian.data.values[0, M:] / \
			(1 + np.linspace(0, 1, M)) ** hurst1
		
		brownian = Brownian(name='fractional')
		brownian.new(1, argvals=np.linspace(0, 2, 2 * M), hurst=hurst2)
		rand_part2 = brownian.data.values[0, M:] / \
			(1 + np.linspace(0, 1, M)) ** hurst2
	
		eps = np.random.normal(0, np.sqrt(0.5), size=M)
		if idx < N1:
			A[idx, :] = h1 + rand_part1 + eps
			B[idx, :] = h3 + 1.5 * rand_part2 + eps
		elif N1 <= idx < N1 + N2:
			A[idx, :] = h2 + rand_part1 + eps
			B[idx, :] = h3 + 0.8 * rand_part2 + eps
		elif N1 + N2 <= idx < N1 + N2 + N3:
			A[idx, :] = h1 + rand_part1 + eps
			B[idx, :] = h3 + 0.2 * rand_part2 + eps
		elif N1 + N2 + N3 <= idx < N1 + N2 + N3 + N4:
			A[idx, :] = h2 + 0.1 * rand_part1 + eps
			B[idx, :] = h2 + 0.2 * rand_part2 + eps
		else:
			A[idx, :] = h3 + rand_part1 + eps
			B[idx, :] = h1 + 0.2 * rand_part2 + eps

	data_1 = DenseFunctionalData({'input_dim_0': np.linspace(0, 1, M)}, A)
	data_2 = DenseFunctionalData({'input_dim_0': np.linspace(0, 1, M)}, B)
	data_fd = MultivariateFunctionalData([data_1, data_2])

	data_1_smooth = data_1.smooth(points=0.5, neighborhood=6)
	data_2_smooth = data_2.smooth(points=0.5, neighborhood=6)
	data_fd_smooth = MultivariateFunctionalData(
		[data_1_smooth, data_2_smooth])

	# Save the data
	#with open(f'./data/scenario_{i}.pkl', 'wb') as f:
	#    pickle.dump(data_fd, f)
	with open(f'./data/scenario_2_{i}_smooth.pkl', 'wb') as f:
		pickle.dump(data_fd_smooth, f)
	with open('./data/labels.pkl', 'wb') as f:
		pickle.dump(labels, f)

	#np.savetxt(f'./data/scenario_{i}_A.csv', A, delimiter=',')
	#np.savetxt(f'./data/scenario_{i}_B.csv', B, delimiter=',')
	np.savetxt(f'./data/scenario_2_{i}_A_smooth.csv', data_1_smooth.values,delimiter=',')
	np.savetxt(f'./data/scenario_2_{i}_B_smooth.csv', data_2_smooth.values,delimiter=',')
	np.savetxt('./data/labels.csv', labels, delimiter=',')

if __name__ == "__main__":
	results = Parallel(n_jobs=num_cores)(delayed(simulation)(i) 
		for i in np.arange(500))
