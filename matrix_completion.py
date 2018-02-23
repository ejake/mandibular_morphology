import numpy as np
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MatrixFactorization


class MatrixCompletion:
    def __init__(self):
        pass

    def fi_complete(self, X, method='mf', **params):
        if method == 'mf':
            #rank = params['rank']=100
            self.X_filled = MatrixFactorization(params['rank']).complete(X)
        if method == 'knn':
            # Use 3 nearest rows which have a feature to fill in each row's missing features
            #k = params['k'] = 3
            self.X_filled = KNN(params['k']).complete(X)
        if method == 'soft':
            # Instead of solving the nuclear norm objective directly, instead
            # induce sparsity using singular value thresholding
            self.X_filled = SoftImpute().complete(X)
        if method == ''

    def error(self, X_gt, type='mse'):
        if hasattr(self, 'X_filled'):
            if type == 'mse':
                nnm_mse = np.linalg.norm(X_gt - self.X_filled) / np.linalg.norm(X_gt)
                print("MF MSE: %f" % nnm_mse)
