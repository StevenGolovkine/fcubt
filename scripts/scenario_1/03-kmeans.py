import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.clustering.fcubt import Node, FCUBT
from joblib import Parallel, delayed

from skfda import FDataGrid
from skfda.ml.clustering import KMeans

from sklearn.metrics import adjusted_rand_score

NUM_CORES = multiprocessing.cpu_count()

def analyze_data(idx):
    argvals = np.loadtxt('./data/argvals.csv')
    values = np.loadtxt(f'./data/scenario_1_{idx}.csv', delimiter=',')
    labels = np.loadtxt('./data/labels.csv')

    data_fd = FDataGrid(values, argvals)

    results_file = {}
    for n_clus in np.arange(2, 9, 1):
        kmeans = KMeans(n_clusters=n_clus)
        final_labels = kmeans.fit_predict(data_fd)
        ARI = adjusted_rand_score(labels, final_labels)
        results_file[n_clus] = ARI
    return results_file

def analyze_data_derivative(idx):
    argvals = np.loadtxt('./data/argvals.csv')
    values = np.loadtxt(f'./data/scenario_1_{idx}.csv', delimiter=',')
    labels = np.loadtxt('./data/labels.csv')

    data_fd = FDataGrid(values, argvals)
    data_fd_deriv = data_fd.derivative(order=1)
    
    results_file = {}
    for n_clus in np.arange(2, 9, 1):
        kmeans = KMeans(n_clusters=n_clus)
        final_labels = kmeans.fit_predict(data_fd_deriv)
        ARI = adjusted_rand_score(labels, final_labels)
        results_file[n_clus] = ARI
    return results_file

def main():
    inputs = range(500)
    
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) 
        for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results_kmeans.pkl", "wb")
    pickle.dump(results, file)
    file.close()

    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data_derivative)(i) 
        for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results_kmeans_derivatives.pkl", "wb")
    pickle.dump(results, file)
    file.close()

if __name__ == '__main__':
    main()
