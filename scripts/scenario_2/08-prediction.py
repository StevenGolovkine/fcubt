import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.simulation import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.clustering.fcubt import Node, FCUBT
from joblib import Parallel, delayed

from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NUM_CORES = multiprocessing.cpu_count()

def analyze_data(idx):
    # Functions
    mean_function_1 = lambda x: 20 / (1 + np.exp(-x))
    mean_function_2 = lambda x: -25 / (1 + np.exp(-x))
    
    # Data generation
    N = 480
    n_features = 3
    n_clusters = 2
    centers = np.array([[0, 0], [0, 0], [0, 0]])
    cluster_std = np.array([[4, 1], [2.66, 0.66], [1.33, 0.33]])

    simu = KarhunenLoeve('wiener', n_functions=n_features)
    simu.new(n_obs=N, n_clusters=n_clusters, centers=centers, cluster_std=cluster_std)

    mean_data = np.vstack([mean_function_1(simu.data.argvals['input_dim_0']),
                           mean_function_2(simu.data.argvals['input_dim_0'])])

    new_values = np.vstack([simu.data.values[:int(N/2), :] + mean_data[0, :],
                            simu.data.values[int(N/2):, :] + mean_data[0, :],
                            simu.data.values[:int(N/2), :] + mean_data[1, :],
                            simu.data.values[int(N/2):, :] + mean_data[1, :],
                            simu.data.values[int(N/2):, :] + mean_data[1, :] - 15 * np.linspace(0, 1, 100)])
    new_labels = np.hstack([simu.labels, simu.labels + 2, np.repeat(4, int(N/2))])
    
    # Train/test split
    values, labels = shuffle(new_values, new_labels, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(values, labels, test_size=5/6, random_state=42)

    data_train = DenseFunctionalData(simu.data.argvals, X_train)
    data_test = DenseFunctionalData(simu.data.argvals, X_test)
    
    # fCUBT
    root_node = Node(data_train, is_root=True)
    fcubt = FCUBT(root_node=root_node)
    fcubt.grow(n_components=0.95)
    fcubt.join(n_components=0.95)
    
    # Prediction
    pred_test = fcubt.predict(data_test, step='join')
    res = []
    for idx in range(1, len(pred_test) + 1):
        res.append(adjusted_rand_score(y_test[:idx], pred_test[:idx]))
    
    return res

def main():
    inputs = range(500)
    
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results_fcubt_pred_200.pkl", "wb")
    pickle.dump(results, file)
    file.close()

if __name__ == '__main__':
    main()