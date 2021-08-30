import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData, MultivariateFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from FDApy.clustering.fcubt import Node, FCUBT
from joblib import Parallel, delayed

from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

NUM_CORES = multiprocessing.cpu_count()

def analyze_data(idx):
    with open(f'./data/scenario_3_{i}_smooth.pkl', 'rb') as f:
        data_fd = pickle.load(f)
    labels = np.loadtxt('./data/labels.csv')
    
    fpca = MFPCA(n_components=[0.99, 0.99])
    fpca.fit(data_fd, method='NumInt')
    scores = fpca.transform()
    
    results_file = {}
    for n_clus in np.arange(2, 9, 1):
        gm = GaussianMixture(n_clus)
        final_labels = gm.fit_predict(scores)
        ARI = adjusted_rand_score(labels, final_labels)
        results_file[n_clus] = ARI
    return results_file

def main():
    inputs = range(500)
    
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) 
        for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results_FPCA_GMM.pkl", "wb")
    pickle.dump(results, file)
    file.close()

if __name__ == '__main__':
    main()