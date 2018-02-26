import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import preprocessing
#from numpy import genfromtxt
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MatrixFactorization


data.data_path = '../../data/'

data = preprocessing.PreprocessingData()
data.run_all()

## Models

X_filled_mf = MatrixFactorization(rank=100).complete(X_incomplete)

## Results

#Correction of 4 nan values, in perfil_angles
#0 6805
#0 6806
#114 5963
#114 5964

perfil_all = np.nan_to_num(perfil_all)
# print mean squared error for the three imputation methods above
nnm_mse = np.linalg.norm(perfil_all- X_filled_mf)/np.linalg.norm(perfil_all)
print("MF MSE: %f" % nnm_mse)

#Write reconstruct matrix
np.savetxt('../data/outcome_mf.csv', X_filled_mf, delimiter=',')
np.savetxt('../data/outcome_mf_class_tr.csv', y_train, delimiter=',')
np.savetxt('../data/outcome_mf_class_ts.csv', y_test, delimiter=',')