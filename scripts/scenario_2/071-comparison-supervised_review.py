import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData, MultivariateFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import UFPCA, MFPCA
from FDApy.clustering.fcubt import Node, FCUBT
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NUM_CORES = multiprocessing.cpu_count()

def analyze_data(idx):
    print(f'Simulation {idx}')
    with open(f'./data/scenario_2_{idx}_smooth.pkl', 'rb') as f:
        data_fd = pickle.load(f)
    labels = np.loadtxt('./data/labels.csv')
    
    # Train/test split
    x = np.arange(data_fd.n_obs)
    np.random.shuffle(x)
    
    data_shuffle = [data[x] for data in data_fd]
    labels_shuffle = labels[x]
    new_data = MultivariateFunctionalData(data_shuffle)
    
    pct = 0.33
    s = int(np.ceil((1 - pct) * new_data.n_obs))
    train = MultivariateFunctionalData([data[:s] for data in new_data])
    test = MultivariateFunctionalData([data[s:] for data in new_data])
    labels_train = labels_shuffle[:s]
    labels_test = labels_shuffle[s:]
        
    # FPCA
    fpca = MFPCA(n_components=[0.99, 0.99])
    fpca.fit(train, method='NumInt')
        
    # Compute scores
    train_proj = fpca.transform(train)
    test_proj = fpca.transform(test)

    # GP classification
    gp = GaussianProcessClassifier(1.0 * RBF(1.0))
    gp.fit(train_proj, labels_train)
    pred_gpc = gp.predict(test_proj)
    ARI_gp = adjusted_rand_score(labels_test, pred_gpc)

    # Random Forest
    clf = RandomForestClassifier()
    clf.fit(train_proj, labels_train)
    pred_cart = clf.predict(test_proj)
    ARI_rf = adjusted_rand_score(labels_test, pred_cart)

    # fCUBT
    root_node = Node(train, is_root=True)
    fcubt = FCUBT(root_node=root_node)
    fcubt.grow(n_components=[0.95, 0.95])
    fcubt.join(n_components=[0.95, 0.95])
    pred_fcubt = fcubt.predict(test, step='join')
    ARI_fcubt = adjusted_rand_score(labels_test, pred_fcubt)
    
    return {'n_clusters': len(np.unique(pred_fcubt)),
            'ARI_gp': ARI_gp,
            'ARI_rf': ARI_rf,
            'ARI_fcubt': ARI_fcubt}

def main():
    inputs = range(500)
    
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) 
        for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results/results_fcubt_classif_review.pkl", "wb")
    pickle.dump(results, file)
    file.close()

if __name__ == '__main__':
    main()