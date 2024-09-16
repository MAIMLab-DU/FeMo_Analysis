import copy
import time
import optuna
import numpy as np
from optuna.samplers import GridSampler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from .base import FeMoBaseClassifier


class FeMoSVClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'C': [10, 20],
                         'gamma': [10, 20],
                         'kernel': ['rbf', 'linear']
                     },
                     'hyperparams': {
                         'C': 1e-3,
                         'gamma': 'auto',
                         'kernel': 'rbf'
                     }
                 }):
        super().__init__(config)

    def _convert_to_logspace(self, array: list):
        return np.logspace(*array).tolist()

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):        
        start = time.time()

        num_folds = len(train_data)

        def callback(study: optuna.Study, trial: optuna.Trial):
            if study.best_trial.number == trial.number:
                study.set_user_attr(key='C', value=trial.user_attrs['C'])
                study.set_user_attr(key='gamma', value=trial.user_attrs['gamma'])

        def objective(trial: optuna.Trial):
            C = trial.suggest_int('C', low=10, high=50, step=10)
            gamma = trial.suggest_int('gamma', low=10, high=50, step=10)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])

            inner_grid = copy.deepcopy(self.search_space)
            hyperparams = copy.deepcopy(self.hyperparams)

            inner_grid.update({'C': np.logspace(-4, 4, C), 
                               'gamma': np.logspace(-4, 4, gamma),
                               'kernel': [kernel]})
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

            
            self.logger.info(f"Performing Grid Search with - "
                            f"{num_folds}x5-fold Cross-validation")
            
            accuracies = []
            best_accuracy = -np.inf
            best_C_ = 1e-4
            best_gamma_ = 1e-4
            for i in range(num_folds):

                hyperparams = self._update_class_weight(train_data[i][:, -1], hyperparams)
                estimator = SVC(random_state=0, **hyperparams)
                cross_validator = GridSearchCV(
                    cv=cv,
                    estimator=estimator,
                    param_grid=inner_grid,
                    n_jobs=-1
                )

                X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
                X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

                cross_validator.fit(X_train, y_train)
                y_pred = cross_validator.predict(X_test)
                accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
                accuracies.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_C_ = cross_validator.best_params_['C']
                    best_gamma_ = cross_validator.best_params_['gamma']

            trial.set_user_attr(key='C', value=best_C_)
            trial.set_user_attr(key='gamma', value=best_gamma_)
            return np.max(accuracies)
        
        # Define search space for GridSampler
        search_space = {
            'C': self.search_space.get('C', [20]),
            'gamma': self.search_space.get('gamma', [20]),
            'kernel': self.search_space.get('kernel', ['rbf', 'linear'])
        }
        sampler = GridSampler(search_space, seed=42)
        n_trials = sampler._n_min_trials

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
        best_params.update(gamma=study.user_attrs['gamma'])
        best_accuracy = study.best_value
        
        self.logger.info(f"Optuna search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_params}")
        self.logger.info(f"Best average model accuracy: {best_accuracy}")

        self.hyperparams.update(
            C=best_params['C'],
            gamma=best_params['gamma'],
            kernel=best_params['kernel']
        )

    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)

        hyperparams = copy.deepcopy(self.hyperparams)
        hyperparams = self._update_class_weight(train_data[0][:, -1], hyperparams)

        self.classifier = SVC(random_state=0, probability=True, **hyperparams)

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
