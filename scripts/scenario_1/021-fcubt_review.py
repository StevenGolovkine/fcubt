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
	with open(f'./data/scenario_1_{idx}_review.pkl', 'rb') as f:
		data_fd = pickle.load(f)
	with open(f'./data/labels_review.pkl', 'rb') as f:
		labels = pickle.load(f)

	root_node = Node(data_fd, is_root=True)
	fcubt = FCUBT(root_node=root_node)
	fcubt.grow(n_components=0.95)
	fcubt.join(n_components=0.95)

	return {'n_clusters': len(np.unique(fcubt.labels_join)),
			'ARI': adjusted_rand_score(labels, fcubt.labels_join)}

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