import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split


class PreprocessingData:
    def __init__(self):
        self.data_path = '../data/'

    def load_perfil_csv(self):
        # load perfil data
        self.perfil_lines = self.check_nan(genfromtxt(self.data_path+'Perfil_Lines.csv', delimiter=','))
        self.perfil_angles = self.check_nan(genfromtxt(self.data_path+'Perfil_Angles.csv', delimiter=','))
        self.perfil_class = self.check_nan(genfromtxt(self.data_path+'Perfil_class.csv', delimiter=',', skip_header=1))
        self.perfil_all = np.concatenate((self.perfil_lines[:, 1:], self.perfil_angles[:, 1:]), axis=1)

    def check_nan(self, x):
        if np.count_nonzero(np.isnan(x)) > 0:
            print 'Count of NaNs replaced:', np.count_nonzero(np.isnan(x))
            print np.argwhere(np.isnan(x))
            #Replace NaNs
            x = np.nan_to_num(x)
        return x

    def split_data(self, test=.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.perfil_all, self.perfil_class,
                                                                                test_size=test, random_state=42)
        self.X_train_lines = self.X_train[:, :self.perfil_lines[:, 1:].shape[1]]
        self.X_test_lines = self.X_test[:, :self.perfil_lines[:, 1:].shape[1]]
        self.X_train_angles = self.X_train[:, self.perfil_lines[:, 1:].shape[1]:]
        self.X_test_angles = self.X_test[:, self.perfil_lines[:, 1:].shape[1]:]


    def load_type_measures(self):
        self.perfil_lines_mandibular_measures = genfromtxt(self.data_path+'Perfil_Lines_type.csv', delimiter=',')
        self.perfil_angles_mandibular_measures = genfromtxt(self.data_path+'Perfil_Angles_type.csv', delimiter=',')

    #before the run this method mandibular measures are marked with a NaN value and non mandibular measures with a non-NaN
    def mask_type_measures(self, mandibular_mask = np.nan, other_mask = 1):
        self.perfil_lines_mask_mandibular_measures = self.perfil_lines_mandibular_measures[1:]
        self.perfil_angles_mask_mandibular_measures = self.perfil_angles_mandibular_measures[1:]

        if mandibular_mask != 0:
            if np.isnan(mandibular_mask):
                # NaN if the measure includes a mandibular landmark, 1 otherwise
                self.perfil_lines_mask_mandibular_measures[~np.isnan(self.perfil_lines_mask_mandibular_measures)] = 1
                self.perfil_angles_mask_mandibular_measures[~np.isnan(self.perfil_angles_mask_mandibular_measures)] = 1
            else:
                self.perfil_lines_mask_mandibular_measures[np.isnan(self.perfil_lines_mask_mandibular_measures)] = mandibular_mask
                self.perfil_angles_mask_mandibular_measures[np.isnan(self.perfil_angles_mask_mandibular_measures)] = mandibular_mask
                self.perfil_lines_mask_mandibular_measures[~np.isnan(self.perfil_lines_mask_mandibular_measures)] = other_mask
                self.perfil_angles_mask_mandibular_measures[~np.isnan(self.perfil_angles_mask_mandibular_measures)] = other_mask

        self.perfil_all_mask_mandibular_measures = np.concatenate((self.perfil_lines_mask_mandibular_measures,
                                                                   self.perfil_angles_mask_mandibular_measures))

        self.X_test_masked = self.X_test * self.perfil_all_mask_mandibular_measures

        # Merge test and train
        self.X_incomplete = np.concatenate((self.X_train, self.X_test_masked))


    def run_all(self):
        self.load_perfil_csv()
        self.split_data()
        self.load_type_measures()
        self.mask_type_measures()

    def load_data_perfil(self):
        self.load_perfil_csv()
        self.load_type_measures()
