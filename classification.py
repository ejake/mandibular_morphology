#Classification pipelines
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

class Classification:
    def __init__(self):
        self.trained = False

    def explore_training(self, X_train, y_train, **params):

        folds = 10
        exploration_tries = 20

        if params['method'] == 'simple_svm':
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
            svc = svm.SVC()
            self.estimator = GridSearchCV(svc, parameters)
            self.estimator.fit(X_train, y_train)

        elif params['method'] == 'simple':
            # PCA components exploratio
            #n_components = np.arange( 10, len(numerical_cols), 2) # number of PCA components
            # Logreg regularization parameter exploration
            #c_s = np.linspace(0.1, 5, num=exploration_tries)
            # GridSearch
            #grid = dict(pca__n_components=n_components, lr__C=c_s)
            # Test with RobustScaler, QuantileTransformer, Normalizer
            #pipe = Pipeline([('scaler',StandardScaler()),('pca', PCA()), ('lr', LR())])

            #kf = StratifiedKFold(n_splits=folds, shuffle=True)
            #estimator = GridSearchCV(pipe, grid, scoring='accuracy', cv=kf, verbose=1, n_jobs=-1)
            #estimator.fit(X_train, y_train)
            pass

        self.trained = True

    def assess_performance(self, X_val, y_val):
        print("\nBest validation f1: {:.3}".format(self.estimator.best_score_))
        print("Best parameters: "+str(self.estimator.best_params_))
        y_pred = self.estimator.predict(X_val)
        print("Accuracy on test set: {:.3}\n".format(metrics.accuracy_score(y_val, y_pred)))
        #print("F1 on test set: {:.3}\n".format(metrics.f1_score(y_val, y_pred)))
        print(metrics.confusion_matrix(y_val, y_pred))