from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial import distance
import numpy as np

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self
    
    def predict(self, X):
        if self.X is None:
            return None
        
        X = np.array(X)
        k = self.n_neighbors

        dists = distance.cdist(self.X, X, metric='euclidean')               # dists_ij = dist(u=XA[i], v=XB[j])
        partition = np.argpartition(dists, kth=k - 1, axis=0)[:k]           # choose for each column k closest
        k_labels = self.y[partition]                                        # choose the labels of the k closest for each point to predict
        predictions = np.sign(np.mean(k_labels, axis=0))                    # calculate the predicted label
        return predictions