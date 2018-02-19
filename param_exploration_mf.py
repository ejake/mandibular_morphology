import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, MatrixFactorization

#load perfil data
perfil_lines = genfromtxt('../data/Perfil_Lines.csv', delimiter=',')
perfil_angles = genfromtxt('../data/Perfil_Angles.csv', delimiter=',')
perfil_class = genfromtxt('../data/Perfil_class.csv', delimiter=',', skip_header=1)

perfil_all = np.concatenate((perfil_lines[:,1:], perfil_angles[:,1:]), axis=1)

X_train, X_test, y_train, y_test = train_test_split(perfil_all, perfil_class, test_size=0.3, random_state=42)

perfil_lines_mandibular_measures = genfromtxt('../data/Perfil_Lines_type.csv', delimiter=',')
perfil_angles_mandibular_measures = genfromtxt('../data/Perfil_Angles_type.csv', delimiter=',')

# 1 if the measure includes a mandibular landmark, 0 otherwise
perfil_lines_mask_mandibular_measures = perfil_lines_mandibular_measures[1:]
perfil_angles_mask_mandibular_measures = perfil_angles_mandibular_measures[1:]
perfil_lines_mask_mandibular_measures[perfil_lines_mask_mandibular_measures > 0]=-1
perfil_angles_mask_mandibular_measures[perfil_angles_mask_mandibular_measures > 0]=-1
perfil_all_mask_mandibular_measures = np.concatenate((perfil_lines_mask_mandibular_measures, \
                                                      perfil_angles_mask_mandibular_measures))

# 0 if the measure includes a mandibular landmark, 1 otherwise
perfil_angles_mask_mandibular_measures[perfil_angles_mask_mandibular_measures == 0] = 1
perfil_angles_mask_mandibular_measures[perfil_angles_mask_mandibular_measures == -1] = 0

perfil_all_mask_mandibular_measures = np.concatenate((perfil_lines_mask_mandibular_measures, \
                                                      perfil_angles_mask_mandibular_measures))

## Preprocessing

# NaN if the measure includes a mandibular landmark, 1 otherwise
perfil_all_mask_mandibular_measures[perfil_all_mask_mandibular_measures == 0] = np.nan
X_test_masked = X_test*perfil_all_mask_mandibular_measures

#Merge test and train
X_incomplete = np.concatenate((X_train,X_test_masked))

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