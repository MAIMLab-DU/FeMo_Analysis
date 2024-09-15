import time
import numpy as np
import copy
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


class FeMoAdaBoostClassifier(FeMoBaseClassifier):

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
                             'n_estimators': [50, 100, 150,
                                              200, 250, 300],
                             'learning_rate': [0.01, 0.1, 0.5,
                                               1.0, 1.5, 2.0]
                         }
                     },
                     'fit_params': {
                         'base_estimator':{
                              'n_estimators': 50,
                              'subsample': 0.75,
                              'max_depth': 1
                         },
                         'n_estimators': 50,
                         'algorithm': 'SAMME.R'
                     }
                 }):
        super().__init__(config)

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):        
        start = time.time()

        num_folds = len(train_data)

        fit_params = copy.deepcopy(self.fit_params)
        fit_params.pop('estimator', None)
        base_estimator = GradientBoostingClassifier(
            random_state=42,
            warm_start=True,
            **fit_params.pop('base_estimator', {})
        )
        estimator = AdaBoostClassifier(
            random_state=0,
            estimator=base_estimator,
            **fit_params
        )        

        search_params = copy.deepcopy(self.search_params)
        cv_params = search_params.pop('cv', {})
        cv = RepeatedStratifiedKFold(random_state=42, **cv_params)

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
                best_model = self.cross_validator.best_estimator_

        best_base_estimator = best_model.estimator_
        n_estimators = best_model.n_estimators
        learning_rate = best_model.learning_rate

        self.logger.info(f"Grid search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model with test accuracy: {best_accuracy: 0.3f}\t"
                         f"n_estimators:{n_estimators}, learning_rate:{learning_rate: 0.3f}")
        
        self.fit_params.update(estimator=best_base_estimator,
                               n_estimators=n_estimators,
                               learning_rate=learning_rate)

    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)

        fit_params = copy.deepcopy(self.fit_params)
        fit_params.pop('base_estimator', None)
        self.classifier = AdaBoostClassifier(random_state=0, **fit_params)

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

                




