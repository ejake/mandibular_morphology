import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import preprocessing
import matrix_completion
#from numpy import genfromtxt
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MatrixFactorization


def explore_param_rank_mf(write_csv = '', list_parameters = range(10, 1000, 10)):
    data = preprocessing.PreprocessingData()
    data.data_path = '../../data/'

    data.run_all()

    ## Models
    completion = matrix_completion.MatrixCompletion()

    mse_list = []
    for i in list_parameters:
        completion.fi_complete(data.X_incomplete, 'mf', rank=i)
        mse_list.append(completion.error(np.nan_to_num(data.perfil_all)))

    if write_csv != '':
        np.savetxt(data.data_path + write_csv, mse_list, delimiter=',')

    return mse_list




## Results

#Correction of 4 nan values, in perfil_angles
#0 6805
#0 6806
#114 5963
#114 5964

#perfil_all = np.nan_to_num(perfil_all)
# print mean squared error for the three imputation methods above
#nnm_mse = np.linalg.norm(perfil_all- X_filled_mf)/np.linalg.norm(perfil_all)
#print("MF MSE: %f" % nnm_mse)

#Write reconstruct matrix
#np.savetxt('../data/outcome_mf.csv', X_filled_mf, delimiter=',')
#np.savetxt('../data/outcome_mf_class_tr.csv', y_train, delimiter=',')
#np.savetxt('../data/outcome_mf_class_ts.csv', y_test, delimiter=',')