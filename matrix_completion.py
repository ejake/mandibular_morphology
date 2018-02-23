import numpy as np
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MatrixFactorization


class MatrixCompletion:
    def __init__(self):
        pass

    def fi_complete(self, X, method='mf', **params):
        if method == 'mf':
            self.X_filled = MatrixFactorization(params['rank']=100).complete(X)
        if method == 'knn':
            # Use 3 nearest rows which have a feature to fill in each row's missing features
            self.X_filled = KNN(params['k']=3).complete(X)

    def error(self, X_gt, type='mse'):
        if hasattr(self, 'X_filled'):
            if type == 'mse':
                nnm_mse = np.linalg.norm(X_gt - self.X_filled) / np.linalg.norm(X_gt)
                print("MF MSE: %f" % nnm_mse)
