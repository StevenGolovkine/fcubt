import multiprocessing
import numpy as np
import pickle
import sys
import time

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
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
    with open(f'./data/scenario_1_{idx}.pkl', 'rb') as f:
        data_fd = pickle.load(f)
    labels = np.loadtxt('./data/labels.csv')
    
    # Train/test split
    values = data_fd.values
    values, labels = shuffle(values, labels, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(values, labels,
                                                        test_size=0.33,
                                                        random_state=42)

    train = DenseFunctionalData(data_fd.argvals, X_train)
    test = DenseFunctionalData(data_fd.argvals, X_test)
    
    # FPCA
    ufpca = UFPCA(n_components=0.99)
    ufpca.fit(data=train, method='GAM')

    scores_train = ufpca.transform(data=train, method='NumInt')
    scores_test = ufpca.transform(data=test, method='NumInt')

    # GP classification
    gp = GaussianProcessClassifier(1.0 * RBF(1.0))
    gp.fit(scores_train, y_train)
    pred_gp = gp.predict(scores_test)
    ARI_gp = adjusted_rand_score(y_test, pred_gp)

    # Random Forest
    clf = RandomForestClassifier()
    clf.fit(scores_train, y_train)
    pred_rf = clf.predict(scores_test)
    ARI_rf = adjusted_rand_score(y_test, pred_rf)

    # fCUBT
    root_node = Node(train, is_root=True)
    fcubt = FCUBT(root_node=root_node)
    fcubt.grow(n_components=0.95)
    fcubt.join(n_components=0.95)
    pred_fcubt = fcubt.predict(test, step='join')
    ARI_fcubt = adjusted_rand_score(y_test, pred_fcubt)
    
    return {'n_clusters': len(np.unique(final_labels)),
            'ARI_gp': ARI_gp,
            'ARI_rf': ARI_rf,
            'ARI_fcubt': ARI_fcubt}

def main():
    inputs = range(500)
    
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(analyze_data)(i) for i in inputs)
    print(f'{time.time() - start}')
    
    file = open("./results_fcubt_classif_review.pkl", "wb")
    pickle.dump(results, file)
    file.close()

if __name__ == '__main__':
    main()