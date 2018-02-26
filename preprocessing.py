import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split


class PreprocessingData:
    def __init__(self):
        self.data_path = '../data/'

    def load_perfil_csv(self):
        # load perfil data
        perfil_lines = genfromtxt(self.data_path+'Perfil_Lines.csv', delimiter=',')
        perfil_angles = genfromtxt(self.data_path+'Perfil_Angles.csv', delimiter=',')
        self.perfil_class = genfromtxt(self.data_path+'Perfil_class.csv', delimiter=',', skip_header=1)
        self.perfil_all = np.concatenate((perfil_lines[:, 1:], perfil_angles[:, 1:]), axis=1)

    def split_data(self, test=.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.perfil_all, self.perfil_class,
                                                                                test_size=test, random_state=42)

    def load_type_measures(self):
        self.perfil_lines_mandibular_measures = genfromtxt(self.data_path+'Perfil_Lines_type.csv', delimiter=',')
        self.perfil_angles_mandibular_measures = genfromtxt(self.data_path+'Perfil_Angles_type.csv', delimiter=',')

    #before the run this method mandibular measures are marked with a value greater than 1 and non mandibular measures with 0
    def mask_type_measures(self, mandibular_mask = np.nan, other_mask = 1):
        # -1 if the measure includes a mandibular landmark, 0 otherwise
        self.perfil_lines_mask_mandibular_measures = self.perfil_lines_mandibular_measures[1:]
        self.perfil_angles_mask_mandibular_measures = self.perfil_angles_mandibular_measures[1:]

        if mandibular_mask != 0:
            if np.isnan(mandibular_mask):
                # NaN if the measure includes a mandibular landmark, 1 otherwise
                self.perfil_lines_mask_mandibular_measures[self.perfil_lines_mask_mandibular_measures > 0] = np.nan
                self.perfil_angles_mask_mandibular_measures[self.perfil_angles_mask_mandibular_measures > 0] = np.nan
                self.perfil_lines_mask_mandibular_measures[self.perfil_lines_mask_mandibular_measures == 0] = 1
                self.perfil_angles_mask_mandibular_measures[self.perfil_angles_mask_mandibular_measures == 0] = 1
            else:
                self.perfil_lines_mask_mandibular_measures[self.perfil_lines_mask_mandibular_measures > 0] = mandibular_mask
                self.perfil_angles_mask_mandibular_measures[self.perfil_angles_mask_mandibular_measures > 0] = mandibular_mask
                self.perfil_lines_mask_mandibular_measures[self.perfil_lines_mask_mandibular_measures == 0] = other_mask
                self.perfil_angles_mask_mandibular_measures[self.perfil_angles_mask_mandibular_measures == 0] = other_mask

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
