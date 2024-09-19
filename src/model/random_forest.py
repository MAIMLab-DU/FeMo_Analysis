import copy
import time
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from .base import FeMoBaseClassifier

optuna.logging.set_verbosity(optuna.logging.ERROR)


class FeMoRFClassifier(FeMoBaseClassifier):

    def __init__(self,
                 config = {
                     'search_space': {
                         'n_estimators': [50, 100, 150],
                         'max_depth': [5, 10],
                         'max_leaf_nodes': [3, 6, 9],
                         'max_features': [
                             'sqrt', 'log2', None,
                             1, 2, 3, 4, 5, 6, 8, 10,
                             12, 14, 16, 17, 19,
                             22, 25, 27, 29
                        ]
                     },
                     'hyperparams': {
                         'n_estimators': 100,
                         'max_depth': None,
                         'max_leaf_nodes': None,
                         'max_features': 'sqrt'
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

                estimator = RandomForestClassifier(
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

        best_accuracy = -1
        best_model = None
        predictions = []
        prediction_scores = []
        accuracy_scores = {
            'train_accuracy': [],
            'test_accuracy': []
        }

        for i in range(num_iterations):
            X_train, y_train = train_data[i][:, :-1], train_data[i][:, -1]
            X_test, y_test = test_data[i][:, :-1], test_data[i][:, -1]

            hyperparams = self._update_class_weight(train_data[i][:, -1], hyperparams)
            estimator = RandomForestClassifier(random_state=0, **hyperparams)
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
