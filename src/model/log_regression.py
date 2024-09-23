import time
import copy
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from .base import FeMoBaseClassifier


class FeMoLogRegClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'C': np.logspace(-4, 4, 20),
                         'solver': ['lbfgs', 'liblinear'],
                         'max_iter': {
                             'low': 500,
                             'high': 1000,
                             'step': 500
                         }
                     },
                     'hyperparams': {
                         'C': 1e-3, 
                         'solver': 'lbfgs',
                         'max_iter': 1000
                     }
                 }):
        super().__init__(config)

    def tune(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray],
               n_trials: int = 10):
        
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)

        def objective(trial: optuna.Trial):

            params = {
                'C': trial.suggest_categorical('C', self.search_space['C']),
                'solver': trial.suggest_categorical('solver', self.search_space['solver']),
                'max_iter': trial.suggest_int('max_iter', **self.search_space['max_iter'])
            }
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)

            accuracy_scores = []
            for i in range(num_folds):
                X_train, y_train = train_data[i][:, :-3], train_data[i][:, -1]

                estimator = LogisticRegression(
                    random_state=42,
                    **params
                )
                cv_score = cross_val_score(
                    estimator=estimator,
                    cv=cv_inner,
                    X=X_train,
                    y=y_train,
                    scoring='accuracy',
                    n_jobs=-1
                )
                accuracy_scores.append(cv_score)
            
            return np.mean(accuracy_scores)
        
        # By default, optuna uses TPE sampling
        study = optuna.create_study(direction='maximize')
        self.logger.info(f"Performing Grid Search with {num_folds}x5 Cross-validation")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_accuracy = study.best_value
        
        self.logger.info(f"Optuna search took {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Best model hyperparameters: {best_params}")
        self.logger.info(f"Best model accuracy: {best_accuracy}")

        for key, value in best_params.items():
            self.hyperparams.update({key: value})

    def fit(self,
            train_data: list[np.ndarray],
            test_data: list[np.ndarray]):
        
        start = time.time()

        num_iterations = len(train_data)
        hyperparams = copy.deepcopy(self.hyperparams)
        
        best_accuracy = -1
        best_model = None
        predictions = []
        prediction_scores = []
        det_indices = []
        filename_hash = []
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-3], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-3], test_data[i][:, -1]

            hyperparams = self._update_class_weight(train_data[i][:, -1], hyperparams)
            estimator = LogisticRegression(verbose=False, **hyperparams)
            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            y_test_pred_score = estimator.predict_proba(X_test)
            predictions.append(y_test_pred)
            prediction_scores.append(y_test_pred_score)

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
                best_model = estimator

            det_indices.append(test_data[i][:, -2])
            filename_hash.append(test_data[i][:, -3])

            self.logger.info(f"Iteration {i+1}:")
            self.logger.info(f"Training Accuracy: {current_train_accuracy:.3f}")
            self.logger.info(f"Test Accuracy: {current_test_accuracy:.3f}")
            self.logger.info(f"Best Test Accuracy: {best_accuracy:.3f}")
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Average training accuracy: {np.mean(accuracy_scores['train_accuracy'])}")
        self.logger.info(f"Average testing accuracy: {np.mean(accuracy_scores['test_accuracy'])}")
        
        self.classifier = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.preds = predictions
        self.result.pred_scores = prediction_scores
        self.result.det_indices = det_indices
        self.result.filename_hash = filename_hash
