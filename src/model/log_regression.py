import time
import copy
import optuna
import numpy as np
from typing import Literal
from optuna.samplers import GridSampler, RandomSampler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from .base import FeMoBaseClassifier


class FeMoLogRegClassifier(FeMoBaseClassifier):

    def __init__(self,
                 search_method: Literal['grid', 'random'] = 'grid',
                 config = {
                     'search_params': {
                         'Cs': [10, 20],
                         'cv': 5,
                         'solver': ['lbfgs', 'liblinear'],
                         'max_iter': 1000,
                         'verbose': False,
                         'n_jobs': 1,
                     },
                     'fit_params': {
                         'solver': 'lbfgs',
                         'max_iter': 1000,
                         'verbose': False,
                         'n_jobs': -1
                     }
                 }):
        super().__init__(config)

        self.search_method = search_method

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_folds = len(train_data)
        search_params = copy.deepcopy(self.search_params)
        search_params = self._update_class_weight(train_data[0][:, -1], search_params)

        def callback(study: optuna.Study, trial: optuna.Trial):
            if study.best_trial.number == trial.number:
                study.set_user_attr(key='C', value=trial.user_attrs['C'])

        def objective(trial: optuna.Trial):
            Cs = trial.suggest_int('Cs', low=10, high=50, step=10)
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])

            search_params.update({'Cs': Cs,'solver': solver})

            cross_validator = LogisticRegressionCV(**search_params)
            self.logger.info(f"Performing Grid Search with - "
                            f"{num_folds}x{self.search_params['cv']}-fold Cross-validation")
            
            accuracies = []
            best_accuracy = -np.inf
            best_C_ = 1e-4
            for i in range(num_folds):
                X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
                X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

                cross_validator.fit(X_train, y_train)
                y_pred = cross_validator.predict(X_test)
                accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
                accuracies.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_C_ = cross_validator.C_[0]

            trial.set_user_attr(key='C', value=best_C_)
            return np.mean(accuracies)
        
        # Define the sampler based on search_method
        if self.search_method == 'grid':
            # Define search space for GridSampler
            search_space = {
                'Cs': self.search_params.get('Cs', [20]),
                'solver': self.search_params.get('solver', ['lbfgs', 'liblinear'])
            }
            sampler = GridSampler(search_space, seed=42)
            n_trials = sampler._n_min_trials
        elif self.search_method == 'random':
            sampler = RandomSampler(seed=42)
            n_trials = 10
        else:
            raise ValueError("Invalid search_method. Use 'grid' or 'random'.")

        # Create an Optuna study with the chosen sampler
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update(C=study.user_attrs['C'])
        best_accuracy = study.best_value
        
        self.logger.info(f"Optuna search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_params}")
        self.logger.info(f"Best average model accuracy: {best_accuracy}")

        self.fit_params.update(
            C=best_params['C'],
            solver=best_params['solver']
        )


    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)
        fit_params = copy.deepcopy(self.fit_params)
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

                




