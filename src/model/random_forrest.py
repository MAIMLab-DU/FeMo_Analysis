import copy
import time
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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

    def search(self, train_data: list[np.ndarray], test_data: list[np.ndarray]):        
        start = time.time()
                
        num_folds = len(train_data)
        
        def objective(trial: optuna.Trial):

            params = {
                name: trial.suggest_categorical(name, value)
                for name, value in self.search_space.items()
            }
            
            outer_scores = []
            for i in range(num_folds):
                X_train_outer, X_test_outer = train_data[i][:, :-1], test_data[i][:, :-1]
                y_train_outer, y_test_outer = train_data[i][:, -1], test_data[i][:, -1]

                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                best_params = None
                best_score = -1

                for train_index_inner, val_index_inner in inner_cv.split(X_train_outer, y_train_outer):
                    X_train_inner, X_val_inner = X_train_outer[train_index_inner], X_train_outer[val_index_inner]
                    y_train_inner, y_val_inner = y_train_outer[train_index_inner], y_train_outer[val_index_inner]

                    inner_params = self._update_class_weight(y_train_inner, params)
                    estimator = RandomForestClassifier(random_state=0, **inner_params) 

                    estimator.fit(X_train_inner, y_train_inner)
                    y_pred_val = estimator.predict(X_val_inner)
                    score = accuracy_score(y_val_inner, y_pred_val) 

                    if score > best_score:
                        best_params = params
                        best_score = score

                # Train the final model on the entire outer training set using the best hyperparameters
                best_params = self._update_class_weight(y_train_outer, best_params)
                estimator = RandomForestClassifier(random_state=0, **best_params)
                estimator.fit(X_train_outer, y_train_outer)
                y_pred_outer = estimator.predict(X_test_outer)
                outer_score = accuracy_score(y_test_outer, y_pred_outer)
                outer_scores.append(outer_score)

            return np.max(outer_scores)
        
        study = optuna.create_study(direction='maximize')
        self.logger.info(f"Performing Grid Search with {num_folds}x5 Cross-validation")
        study.optimize(objective, n_trials=10, show_progress_bar=True)

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
        hyperparams = self._update_class_weight(train_data[0][:, -1], hyperparams)

        self.classifier = RandomForestClassifier(random_state=0, **hyperparams)

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
