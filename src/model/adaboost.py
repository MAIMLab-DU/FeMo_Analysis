import time
import copy
import optuna
import numpy as np
from .base import FeMoBaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score


class FeMoAdaBoostClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_params': {
                         'n_estimators': [50, 100, 150,
                                          200, 250, 300],
                         'learning_rate': [0.01, 0.1, 0.5,
                                           1.0, 1.5, 2.0]
                     },
                     'fit_params': {
                         'n_estimators': 50,
                         'learning_rate': 0.1
                     }
                 }):
        super().__init__(config)

    def search(self,
               train_data: list[np.ndarray],
               test_data: list[np.ndarray],
               n_trials: int = 10):
                
        start = time.time()

        assert len(train_data) == len(test_data), "Train, test data must have same folds"
        num_folds = len(train_data)
        
        def objective(trial: optuna.Trial):

            params = {
                name: trial.suggest_categorical(name, value)
                for name, value in self.search_space.items()
            }
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)

            accuracy_scores = []
            for i in range(num_folds):
                X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]

                estimator = AdaBoostClassifier(
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

    def fit(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):
        start = time.time()

        num_iterations = len(train_data)

        hyperparams = copy.deepcopy(self.hyperparams)

        best_accuracy = -np.inf
        best_model = None
        predictions = []
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

            estimator = AdaBoostClassifier(random_state=0, **hyperparams)
            estimator.fit(X_train, y_train)

            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            predictions.append(y_test_pred)

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

            self.logger.info(f"Iteration {i+1}:")
            self.logger.info(f"Training Accuracy: {current_train_accuracy:.3f}")
            self.logger.info(f"Test Accuracy: {current_test_accuracy:.3f}")
            self.logger.info(f"Best Test Accuracy: {best_accuracy:.3f}")
        
        self.logger.info(f"Fitting model with train data took: {time.time() - start: 0.2f} seconds")
        self.logger.info(f"Average training accuracy: {np.mean(accuracy_scores['train_accuracy'])}")
        self.logger.info(f"Average testing accuracy: {np.mean(accuracy_scores['test_accuracy'])}")
        
        self.classifier = best_model
        self.result.accuracy_scores = accuracy_scores
        self.result.predictions = predictions
