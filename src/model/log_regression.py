import time
import numpy as np
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


class FeMoLogRegClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_params': {
                         'Cs': 20,
                         'cv': 5,
                         'penalty': 'l2',
                         'solver': 'lbfgs',
                         'max_iter': 1000,
                         'verbose': False,
                         'n_jobs': -1,
                     },
                     'fit_params': {
                         'solver': 'lbfgs',
                         'max_iter': 1000,
                         'verbose': False,
                         'n_jobs': -1
                     }
                 }):
        super().__init__(config)

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_folds = len(train_data)
        search_params = self.search_params.copy()
        search_params = self._update_class_weight(train_data[0][:, -1], search_params)

        self.cross_validator = LogisticRegressionCV(**search_params)

        best_accuracy = -np.inf
        self.logger.info(f"Performing Grid Search with - "
                         f"{num_folds}x{self.search_params['cv']}-fold Cross-validation")
        
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
                self.fit_params.update(C=self.cross_validator.C_[0])

        self.logger.info(f"Grid search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model with test accuracy: {best_accuracy}\tC:{self.cross_validator.C_[0]}")


    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)
        fit_params = self.fit_params.copy()
        fit_params = self._update_class_weight(train_data[0][:, -1], fit_params)

        self.classifier = LogisticRegression(**fit_params)
        
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
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.3f} seconds")

        self.result.accuracy_scores = accuracy_scores
        self.result.best_model = best_model
        self.result.best_model_hyperparams = best_model.get_params()

                




