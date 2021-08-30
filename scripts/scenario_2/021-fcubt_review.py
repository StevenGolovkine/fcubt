import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.clustering.fcubt import Node, FCUBT
from joblib import Parallel, delayed

from sklearn.metrics import adjusted_rand_score

NUM_CORES = multiprocessing.cpu_count()

def analyze_data(idx):
	print(f'Simulation {idx}')
	with open(f'./data/scenario_2_{idx}_smooth.pkl', 'rb') as f:
		data_fd = pickle.load(f)
	labels = np.loadtxt('./data/labels.csv')

	# Split the root node
	root_node = Node(data_fd, is_root=True)
	root_node.split(n_components=[4, 4])

	# Build the tree for the left/right node
	fcubt_left = FCUBT(root_node=root_node.left)
	fcubt_right = FCUBT(root_node=root_node.right)

	fcubt_left.grow(n_components=[2, 2], min_size=10)
	fcubt_left.join(n_components=[2, 2])

	fcubt_right.grow(n_components=[2, 2], min_size=10)
	fcubt_right.join(n_components=[2, 2])

	labels_pred = np.concatenate(
		[fcubt_left.labels_join, 
		 fcubt_right.labels_join + np.max(fcubt_left.labels_join) + 1]
	)

	return {'n_clusters': len(np.unique(labels_pred)),
			'ARI': adjusted_rand_score(labels, labels_pred)}

def main():
	inputs = range(500)
	
	start = time.time()
	results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) 
		for i in inputs)
	print(f'{time.time() - start}')
	
	file = open("./results/results_fcubt_review.pkl", "wb")
	pickle.dump(results, file)
	file.close()

if __name__ == '__main__':
	main()