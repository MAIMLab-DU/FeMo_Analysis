import copy
import time
import numpy as np
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


class FeMoRFClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_params': {
                         'cv': {
                             'n_splits': 5,
                             'n_repeats': 1
                         },
                         'scoring': 'accuracy',
                         'verbose': 0,
                         'n_jobs': -1,
                         'param_grid': {
                             'max_features': [
                                 'sqrt', 'log2', None,
                                 1, 2, 3, 4, 5, 6, 8, 10,
                                 12, 14, 16, 17, 19,
                                 22, 25, 27, 29
                             ]
                         }
                     },
                     'fit_params': {
                         'n_estimators': 100,
                         'min_samples_leaf': 50,
                         'n_jobs': -1,
                         'verbose': 0
                     }
                 }):
        super().__init__(config)

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):        
        start = time.time()

        num_folds = len(train_data)

        search_params = copy.deepcopy(self.search_params)
        cv_params = search_params.pop('cv', {})
        cv = RepeatedStratifiedKFold(random_state=42, **cv_params)

        fit_params = copy.deepcopy(self.fit_params)
        fit_params = self._update_class_weight(train_data[0][:, -1], fit_params)
        estimator = RandomForestClassifier(random_state=0, **fit_params)

        self.cross_validator = GridSearchCV(cv=cv,
                                            estimator=estimator,
                                            **search_params)

        best_accuracy = -np.inf
        best_model = None
        self.logger.info(f"Performing Grid Search with - "
                         f"{num_folds}x{cv_params['n_splits']}-fold Cross-validation")
        
        for i in range(num_folds):
            X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]

            self.cross_validator.fit(X_train, y_train)

            y_pred = self.cross_validator.predict(test_data[i][:, :-1])
            current_accuracy = accuracy_score(
                y_pred=y_pred,
                y_true=test_data[i][:, -1]
            )

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_model = self.cross_validator.best_estimator_.estimators_[0]

        self.logger.info(f"Grid search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model with test accuracy: {best_accuracy: 0.3f}\t"
                         f"min_samples_leaf: {best_model.min_samples_leaf}, "
                         f"max_features: {best_model.max_features}")
        
        self.fit_params.update(min_samples_leaf=best_model.min_samples_leaf,
                               max_features=best_model.max_features)

    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)

        fit_params = copy.deepcopy(self.fit_params)
        fit_params = self._update_class_weight(train_data[0][:, -1], fit_params)

        self.classifier = RandomForestClassifier(random_state=0, **fit_params)

        best_accuracy = -np.inf
        best_model = None
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

            self.classifier.fit(X_train, y_train)

            y_train_pred = self.classifier.predict(X_train)
            y_test_pred = self.classifier.predict(X_test)
            current_train_accuracy = accuracy_score(
                y_pred=y_train_pred,
                y_true=y_train
            )
            current_test_accuracy = accuracy_score(
                y_pred=y_test_pred,
                y_true=y_test
            )
            accuracy_scores['train_accuracy'].append(current_train_accuracy)
            accuracy_scores['test_accuracy'].append(current_test_accuracy)

            if current_test_accuracy > best_accuracy:
                best_accuracy = current_test_accuracy
                best_model = self.classifier
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.result.accuracy_scores = accuracy_scores
        self.result.best_model = best_model
        self.result.best_model_hyperparams = best_model.get_params()

                




